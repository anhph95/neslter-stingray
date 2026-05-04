#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tator
# =======================
# ====== HELPERS ========
# =======================
def log(msg):
    """Simple timestamped logger."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
def normalize_name(name: str) -> str:
    """Normalize media names (handles .avi inconsistencies + casing)."""
    if pd.isna(name):
        return None
    return Path(str(name)).stem.lower()
# =======================
# ====== MAIN ===========
# =======================
def main():
    parser = argparse.ArgumentParser(
        description="Add Tator annotation links to a precomputed media/frame CSV."
    )
    parser.add_argument("--csv-in", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--host", default="https://stingray.tator.whoi.edu")
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    t0 = time.time()
    # =======================
    # Load CSV
    # =======================
    log(f"Loading CSV: {args.csv_in}")
    df = pd.read_csv(args.csv_in)
    if "media" not in df.columns or "frame" not in df.columns:
        raise ValueError("CSV must contain 'media' and 'frame' columns.")
    log(f"Rows loaded: {len(df):,}")
    df["media_norm"] = df["media"].apply(normalize_name)
    unique_media = df["media_norm"].dropna().nunique()
    log(f"Unique media (normalized): {unique_media:,}")
    # =======================
    # Fetch Tator media
    # =======================
    log("Connecting to Tator API...")
    t_api = time.time()
    api = tator.get_api(host=args.host, token=args.token)
    project_objs = sorted(
        api.get_media_list(args.project_id, dtype="video"),
        key=lambda p: p.id,
    )
    log(f"Tator media fetched: {len(project_objs):,} (took {time.time() - t_api:.2f}s)")
    # =======================
    # Build mapping
    # =======================
    log("Building media name → ID map")
    media_map = {}
    duplicates = 0
    for obj in project_objs:
        norm = normalize_name(obj.name)
        if not norm:
            continue
        if norm in media_map:
            duplicates += 1
            continue
        media_map[norm] = obj.id
    log(f"Unique normalized Tator names: {len(media_map):,}")
    if duplicates:
        log(f"Duplicate normalized names skipped: {duplicates}")
    # =======================
    # Map IDs
    # =======================
    log("Mapping media to Tator IDs")
    df["id"] = df["media_norm"].map(media_map)
    matched_rows = df["id"].notna().sum()
    unmatched_rows = len(df) - matched_rows
    log(f"Matched rows: {matched_rows:,}")
    log(f"Unmatched rows: {unmatched_rows:,}")
    unmatched_media = df.loc[df["id"].isna(), "media"].dropna().unique()
    if len(unmatched_media) > 0:
        sample = unmatched_media[:10]
        log(f"Unmatched media (sample up to 10): {list(sample)}")
        log(f"Total unmatched unique media: {len(unmatched_media):,}")
    # =======================
    # Build links (vectorized)
    # =======================
    log("Generating annotation links")
    df["link"] = np.nan
    mask = df["id"].notna() & df["frame"].notna()
    df.loc[mask, "link"] = (
        args.host.rstrip("/")
        + "/"
        + str(args.project_id)
        + "/annotation/"
        + df.loc[mask, "id"].astype(int).astype(str)
        + "?frame="
        + df.loc[mask, "frame"].astype(int).astype(str)
    )
    link_count = df["link"].notna().sum()
    log(f"Links generated: {link_count:,}")
    # =======================
    # Cleanup + write
    # =======================
    df.drop(columns=["media_norm"], inplace=True)
    log(f"Writing output: {args.csv_out}")
    df.to_csv(args.csv_out, index=False)
    # =======================
    # Summary
    # =======================
    total_time = time.time() - t0
    log("==== SUMMARY ====")
    log(f"Total rows: {len(df):,}")
    log(f"Links created: {link_count:,}")
    log(f"Coverage: {link_count / len(df) * 100:.2f}%")
    log(f"Elapsed time: {total_time:.2f}s")
if __name__ == "__main__":
    main()