#!/usr/bin/env python3

import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

# =======================
# ====== HELPERS ========
# =======================

def list_files(directory):
    """Recursively lists all files in a directory."""
    file_list = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                file_list.append(entry.path)
            elif entry.is_dir():
                file_list.extend(list_files(entry.path))
    return file_list

def process_video(file_path):
    """Extract per-frame metadata."""
    media_name = Path(file_path).stem
    try:
        base_time = datetime.strptime(
            media_name.split('-')[-1].rstrip('Z'),
            "%Y%m%dT%H%M%S.%f"
        )
    except ValueError:
        print(f"Warning: could not parse datetime from {media_name}")
        base_time = None

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Warning: cannot open {file_path}")
        return [{
            "media_path": file_path,
            "media": media_name,
            "media_time": base_time,
            "frame": None,
            "times": None,
            "status": "bad_file"
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
                "status": "bad_frame"
            })
        else:
            timestamp = base_time + timedelta(milliseconds=ms) if base_time else None
            records.append({
                "media_path": file_path,
                "media": media_name,
                "media_time": base_time,
                "frame": frame_idx,
                "times": timestamp,
                "status": "ok"
            })

        frame_idx += 1

    cap.release()
    return records

def build_dataframe(media_dir, max_workers, limit=None):
    file_paths = list_files(media_dir)
    if limit:
        file_paths = file_paths[:limit]

    all_records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for records in tqdm(executor.map(process_video, file_paths),
                            total=len(file_paths),
                            desc="Processing videos"):
            all_records.extend(records)

    return pd.DataFrame(all_records)

# =======================
# ====== MAIN ===========
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", required=True, help="Directory containing videos")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files (debugging)")
    args = parser.parse_args()

    df = build_dataframe(args.media_dir, args.max_workers, args.limit)
    df.sort_values(["media", "frame"], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Saved timestamps to {args.out}")

if __name__ == "__main__":
    main()
