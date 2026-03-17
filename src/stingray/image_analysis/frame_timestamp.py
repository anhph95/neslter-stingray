#!/usr/bin/env python3
import os
import cv2
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# =======================
# ====== DEFAULTS ======
# =======================
DEFAULT_BASE_MEDIA_DIR = "/proj/nes-lter/Stingray/data"
DEFAULT_MAX_WORKERS = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) - 1)
VIDEO_TYPE_CONFIG = {
    "Basler_a2a2840-14gmBAS": {
        "fps": 12.0,
        "out_dir": "/proj/nes-lter/Stingray/data/media_list/ISIIS2",
    },
    "Basler_avA2300-25gm": {
        "fps": 15.0,
        "out_dir": "/proj/nes-lter/Stingray/data/media_list/ISIIS1",
    },
}
DEFAULT_SUFFIXES = {".avi", ".mp4", ".png", ".tiff"}
# =======================
# ====== HELPERS ========
# =======================
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
def normalize_suffixes(suffixes):
    return {
        s.lower() if str(s).startswith(".") else f".{str(s).lower()}"
        for s in suffixes
    }
def list_files(directory):
    file_list = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                file_list.append(entry.path)
            elif entry.is_dir():
                file_list.extend(list_files(entry.path))
    return file_list
def parse_media_time(media_name):
    try:
        return datetime.strptime(
            media_name.split("-")[-1].rstrip("Z"),
            "%Y%m%dT%H%M%S.%f",
        )
    except ValueError:
        return pd.NaT
def get_file_size(file_path):
    try:
        return (file_path, os.stat(file_path).st_size)
    except FileNotFoundError:
        return (file_path, None)
    except Exception:
        return (file_path, None)
def get_frame_count(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix in {".png", ".tiff", ".tif", ".jpg", ".jpeg"}:
        return 1
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count
def count_one_file(file_path):
    frame_count = get_frame_count(file_path)
    return (file_path, frame_count)
def resolve_config(video_type, fps_arg, out_dir_arg):
    if video_type not in VIDEO_TYPE_CONFIG:
        raise ValueError(
            f"Unknown --video-type '{video_type}'. "
            f"Known types: {', '.join(VIDEO_TYPE_CONFIG)}"
        )
    config = VIDEO_TYPE_CONFIG[video_type]
    fps = fps_arg if fps_arg is not None else config["fps"]
    out_dir = out_dir_arg if out_dir_arg is not None else config["out_dir"]
    return fps, out_dir
def build_base_dataframe(media_dir, max_workers, suffixes=None, file_limit=None):
    file_paths = list_files(media_dir)
    allowed = normalize_suffixes(suffixes) if suffixes else DEFAULT_SUFFIXES
    file_paths = [f for f in file_paths if Path(f).suffix.lower() in allowed]
    if file_limit:
        file_paths = file_paths[:file_limit]
    if not file_paths:
        return pd.DataFrame()
    suffix_counts = {}
    for f in file_paths:
        suf = Path(f).suffix.lower()
        suffix_counts[suf] = suffix_counts.get(suf, 0) + 1
    log(f"Files found after suffix filter: {len(file_paths):,}")
    log(f"Suffix counts: {suffix_counts}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_list_with_sizes = list(
            tqdm(
                executor.map(get_file_size, file_paths),
                total=len(file_paths),
                desc=f"Stat {Path(media_dir).name}",
            )
        )
    df = pd.DataFrame(file_list_with_sizes, columns=["media_path", "media_size"])
    df["media"] = df["media_path"].apply(lambda x: Path(x).stem)
    df["media_time"] = df["media"].apply(parse_media_time)
    return df
def assign_frame_counts_fast(df, max_workers):
    df = df.copy()
    valid_sizes = df["media_size"].dropna()
    if valid_sizes.empty:
        log("No valid file sizes found.")
        df["frame_count"] = None
        return df
    standard_size = valid_sizes.mode().iloc[0]
    ref_candidates = df.loc[df["media_size"] == standard_size, "media_path"]
    if ref_candidates.empty:
        log("No reference file found for modal file size.")
        df["frame_count"] = None
        return df
    reference_file = ref_candidates.iloc[0]
    log(f"Fast mode modal file size: {standard_size}")
    log(f"Reference file: {reference_file}")
    reference_count = get_frame_count(reference_file)
    log(f"Reference frame count: {reference_count}")
    df["frame_count"] = reference_count
    odd_mask = df["media_size"] != standard_size
    odd_files = df.loc[odd_mask, "media_path"].tolist()
    log(f"Odd-sized files requiring frame count check: {len(odd_files):,}")
    if odd_files:
        with ThreadPoolExecutor(max_workers=min(4, max_workers)) as executor:
            results = list(
                tqdm(
                    executor.map(get_frame_count, odd_files),
                    total=len(odd_files),
                    desc="Odd file frames",
                )
            )
        df.loc[odd_mask, "frame_count"] = results
    missing_counts = int(df["frame_count"].isna().sum())
    if missing_counts:
        log(f"Files with missing frame_count: {missing_counts:,}")
    return df
def expand_frames(df, fps):
    df = df.copy()
    before_rows = len(df)
    df = df.dropna(subset=["frame_count"])
    df = df[df["frame_count"] > 0].copy()
    log(f"Files retained for frame expansion: {len(df):,} / {before_rows:,}")
    if df.empty:
        return df
    df["frame_count"] = df["frame_count"].astype(int)
    total_frames = int(df["frame_count"].sum())
    log(f"Total frames to expand: {total_frames:,}")
    df = df.loc[df.index.repeat(df["frame_count"])].copy()
    df["frame"] = df.groupby("media_path").cumcount()
    df["times"] = df["media_time"] + pd.to_timedelta(df["frame"] / fps, unit="s")
    return df
def process_media_details(file_path):
    """
    Old-script behavior for details mode:
    - videos: use per-frame CAP_PROP_POS_MSEC
    - images: emit one frame with media_time as timestamp
    """
    media_name = Path(file_path).stem
    base_time = parse_media_time(media_name)
    suffix = Path(file_path).suffix.lower()
    if suffix in {".png", ".tiff", ".tif", ".jpg", ".jpeg"}:
        return [{
            "media_path": file_path,
            "media": media_name,
            "media_time": base_time,
            "frame": 0,
            "times": base_time if pd.notna(base_time) else None,
            "status": "ok",
        }]
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return [{
            "media_path": file_path,
            "media": media_name,
            "media_time": base_time,
            "frame": None,
            "times": None,
            "status": "bad_file",
        }]
    records = []
    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms <= 0:
            records.append({
                "media_path": file_path,
                "media": media_name,
                "media_time": base_time,
                "frame": frame_idx,
                "times": None,
                "status": "bad_frame",
            })
        else:
            timestamp = base_time + timedelta(milliseconds=ms) if pd.notna(base_time) else None
            records.append({
                "media_path": file_path,
                "media": media_name,
                "media_time": base_time,
                "frame": frame_idx,
                "times": timestamp,
                "status": "ok",
            })
        frame_idx += 1
    cap.release()
    return records
def extract_details_dataframe(media_dir, max_workers, suffixes=None, file_limit=None):
    file_paths = list_files(media_dir)
    allowed = normalize_suffixes(suffixes) if suffixes else DEFAULT_SUFFIXES
    file_paths = [f for f in file_paths if Path(f).suffix.lower() in allowed]
    if file_limit:
        file_paths = file_paths[:file_limit]
    if not file_paths:
        return pd.DataFrame()
    suffix_counts = {}
    for f in file_paths:
        suf = Path(f).suffix.lower()
        suffix_counts[suf] = suffix_counts.get(suf, 0) + 1
    log(f"Files found after suffix filter: {len(file_paths):,}")
    log(f"Suffix counts: {suffix_counts}")
    log(f"Running full per-frame timestamp extraction for {len(file_paths):,} files")
    all_records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for records in tqdm(
            executor.map(process_media_details, file_paths),
            total=len(file_paths),
            desc="Details extraction",
        ):
            all_records.extend(records)
    df = pd.DataFrame(all_records)
    if not df.empty:
        bad_files = int((df["status"] == "bad_file").sum()) if "status" in df.columns else 0
        bad_frames = int((df["status"] == "bad_frame").sum()) if "status" in df.columns else 0
        log(f"Bad file rows: {bad_files:,}")
        log(f"Bad frame rows: {bad_frames:,}")
    return df
# =======================
# ====== MAIN ===========
# =======================
def main():
    parser = argparse.ArgumentParser(
        description="Build media CSV using fast modal-size logic or full per-frame timestamp extraction."
    )
    parser.add_argument("--cruise", required=True, help="Cruise to process")
    parser.add_argument("--video-type", required=True, choices=list(VIDEO_TYPE_CONFIG.keys()))
    parser.add_argument("--base-media-dir", default=DEFAULT_BASE_MEDIA_DIR)
    parser.add_argument("--out-dir", default=None, help="Override default output directory for the selected video type")
    parser.add_argument("--fps", type=float, default=None, help="Override default fps for the selected video type")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--file-limit", type=int, default=None)
    parser.add_argument(
        "--suffix",
        nargs="+",
        default=None,
        help="Restrict to file suffix(es), e.g. --suffix .avi .mp4",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Use full per-frame timestamp extraction instead of fast modal file-size shortcut",
    )
    args = parser.parse_args()
    t0 = time.time()
    fps, out_dir = resolve_config(args.video_type, args.fps, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cruise = args.cruise
    media_dir = f"{args.base_media_dir}/NESLTER_{cruise}/{args.video_type}"
    log(f"Processing cruise: {cruise}")
    log(f"Video type: {args.video_type}")
    log(f"Media dir: {media_dir}")
    log(f"FPS: {fps}")
    log(f"Output dir: {out_dir}")
    log(f"Max workers: {args.max_workers}")
    log(f"Mode: {'details' if args.details else 'fast'}")
    allowed_suffixes = normalize_suffixes(args.suffix) if args.suffix else DEFAULT_SUFFIXES
    log(f"Allowed suffixes: {sorted(allowed_suffixes)}")
    if args.details:
        df_out = extract_details_dataframe(
            media_dir=media_dir,
            max_workers=args.max_workers,
            suffixes=args.suffix,
            file_limit=args.file_limit,
        )
        mode_name = "details"
    else:
        df = build_base_dataframe(
            media_dir=media_dir,
            max_workers=args.max_workers,
            suffixes=args.suffix,
            file_limit=args.file_limit,
        )
        if df.empty:
            log(f"No files found in {media_dir}")
            return
        log(f"Rows in base dataframe: {len(df):,}")
        df = assign_frame_counts_fast(df, args.max_workers)
        df_out = expand_frames(df, fps)
        mode_name = "fast"
    if df_out.empty:
        log(f"No frame data generated for {cruise}")
        return
    sort_cols = ["media", "frame"] if "frame" in df_out.columns else ["media"]
    df_out.sort_values(sort_cols, inplace=True)
    valid_times = df_out["times"].dropna() if "times" in df_out.columns else pd.Series(dtype="datetime64[ns]")
    datestr = valid_times.iloc[0].strftime("%Y%m%d") if not valid_times.empty else cruise
    out_file = f"{out_dir}/{datestr}_{cruise}_{mode_name}.csv"
    df_out.to_csv(out_file, index=False)
    log(f"Saved: {out_file}")
    log(f"Output rows: {len(df_out):,}")
    if "status" in df_out.columns:
        log(f"Status counts: {df_out['status'].value_counts(dropna=False).to_dict()}")
    log(f"Elapsed time: {time.time() - t0:.2f}s")
if __name__ == "__main__":
    main()