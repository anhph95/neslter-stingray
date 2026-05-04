#!/usr/bin/env python3

import argparse
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from numba import njit
from scipy.stats import poisson
from utils import convert_timestamp

# =========================
# --- Utils / Binning ---
# =========================

@njit
def assign_time_bins(timestamps, bin_width, grid_start, grid_end):
    n = timestamps.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_bins = int(np.ceil((grid_end - grid_start) / bin_width))

    for i in range(n):
        ti = timestamps[i]
        if not np.isfinite(ti):
            out[i] = np.nan
            continue

        idx = int(np.floor((ti - grid_start) / bin_width))
        if idx < 0 or idx >= n_bins:
            out[i] = np.nan
        else:
            out[i] = grid_start + idx * bin_width

    return out


def poisson_count_ci(count, ci=0.95):
    if count < 0:
        raise ValueError("Counts must be non-negative.")
    lower, upper = poisson.interval(ci, count)
    return lower, upper


def add_poisson_ci(df_bin_agg, raw_counts_by_bin, data_cols, scale_factor=1.0, ci=0.95):
    ci_results = {}

    for col in data_cols:
        lowers, uppers = [], []
        for count in raw_counts_by_bin[col]:
            lo, hi = poisson_count_ci(int(count), ci=ci)
            lowers.append(lo * scale_factor)
            uppers.append(hi * scale_factor)

        ci_results[f'{col}_ci_lower'] = lowers
        ci_results[f'{col}_ci_upper'] = uppers

    ci_df = pd.DataFrame(ci_results, index=df_bin_agg.index)
    return pd.concat([df_bin_agg, ci_df], axis=1)


# =========================
# --- Main Pipeline ---
# =========================

def main(args):

    print("Loading YOLO CSV...")
    df = pd.read_csv(args.yolo_csv, sep=" ")

    print("Loading class dictionary...")
    with open(args.data_yaml) as f:
        data = yaml.safe_load(f)
    class_dict = data["names"]

    # --- Extract media + frame ---
    df[['media', 'frame']] = df['filename'].str.extract(r'^(.*)_(\d+)\.txt$')
    df['frame'] = df['frame'].astype(int)

    # --- Match original column names ---
    df['score'] = df['confidence']
    df['x'] = df['x_center']
    df['y'] = df['y_center']
    df['class_idx'] = df['class_id'].astype(int)
    df['class'] = df['class_idx'].map(class_dict)

    df = df[['media', 'frame', 'x', 'y', 'width', 'height', 'score', 'class_idx', 'class']]
    df = df.sort_values(['media', 'frame', 'x', 'y']).reset_index(drop=True)

    # --- Geometry ---
    w, h = args.image_width, args.image_height
    um_per_pixel = args.um_per_pixel

    df['area'] = df['width'] * df['height'] * w * h * um_per_pixel * 0.001
    df['size'] = np.maximum(df['width'] * w, df['height'] * h) * um_per_pixel * 0.001

    df = df[df['score'] >= args.score_thresh].reset_index(drop=True)

    # --- Counts ---
    df_count = df.groupby(['media', 'frame', 'class']).size().unstack(fill_value=0).reset_index()
    data_cols = df_count.columns.difference(['media', 'frame', 'total_abundance']).to_list()

    # --- Load sensor + media ---
    print("Loading sensor + media CSVs...")
    sensor_df = pd.read_csv(args.sensor_csv)
    media_df = pd.read_csv(args.media_csv)

    abundance_df = pd.merge(media_df[['times', 'media', 'frame']], df_count, on=['media', 'frame'], how='left')
    abundance_df[data_cols] = abundance_df[data_cols].fillna(0)

    sensor_df['times'] = pd.to_datetime(sensor_df['times'], errors='coerce')
    abundance_df['times'] = pd.to_datetime(abundance_df['times'], errors='coerce')

    sensor_df = sensor_df.sort_values('times')
    abundance_df = abundance_df.sort_values('times')

    ORIGIN = datetime(1904, 1, 1)
    sensor_df["timestamp"] = (sensor_df["times"] - ORIGIN).dt.total_seconds()
    abundance_df["timestamp"] = (abundance_df["times"] - ORIGIN).dt.total_seconds()

    # --- Binning ---
    abundance_df["timestamp"] = assign_time_bins(
        abundance_df["timestamp"].values,
        bin_width=args.bin_width,
        grid_start=sensor_df["timestamp"].min(),
        grid_end=sensor_df["timestamp"].max()
    )

    abundance_df["times"] = pd.to_datetime(convert_timestamp(abundance_df["timestamp"])[0])

    # --- Aggregation ---
    df_bin = abundance_df.groupby("times", as_index=False).agg({
        'times': 'first',
        'media': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
        'frame': lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
        **{col: 'mean' for col in data_cols}
    })

    frame_num = abundance_df.groupby("times").size().rename("frame_num").reset_index()
    df_bin = df_bin.merge(frame_num, on="times", how="left")

    # --- Scale to indiv / m^3 ---
    scale_factor = 1 / args.volume_per_frame
    df_bin[data_cols] = df_bin[data_cols] * scale_factor
    df_bin["total_abundance"] = df_bin[data_cols].sum(axis=1)

    # --- Optional CI ---
    if args.add_ci:
        print("Computing Poisson confidence intervals...")
        raw_counts_by_bin = (
            abundance_df.groupby("times")[data_cols]
            .sum()
            .reset_index()
            .set_index("times")
            .reindex(df_bin["times"])
            .fillna(0)
            .astype(int)
        )
        df_bin = add_poisson_ci(df_bin, raw_counts_by_bin, data_cols, scale_factor=scale_factor)

    # --- Merge with sensor ---
    df_merged = pd.merge(sensor_df, df_bin.drop(columns=['media','frame'], errors='ignore'),
                         on="times", how="left").sort_values("times").reset_index(drop=True)

    print(f"Writing output: {args.out_csv}")
    df_merged.to_csv(args.out_csv, index=False)
    print("Done ✅")


# =========================
# --- CLI ---
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO → Shadowgraph abundance time-binned CLI")

    parser.add_argument("--yolo_csv", default="ar88_yolo_concatenated_results.csv")
    parser.add_argument("--data_yaml", default="/mnt/vast/omics/sosik/yolozone/training_data_v3_20251113/data.yaml")
    parser.add_argument("--sensor_csv", default="~/workdir/neslter-stingray/dash_data/data/stingray_timebinned/20250424_AR88.csv")
    parser.add_argument("--media_csv", default="~/workdir/neslter-stingray/media_list/ISIIS1/20250418_AR88.csv")
    parser.add_argument("--out_csv", default="20250424_AR88_shadowgraph.csv")

    parser.add_argument("--score_thresh", type=float, default=0.7)
    parser.add_argument("--bin_width", type=float, default=5.0, help="seconds")
    parser.add_argument("--image_width", type=int, default=2330)
    parser.add_argument("--image_height", type=int, default=1750)
    parser.add_argument("--um_per_pixel", type=float, default=40.0)
    parser.add_argument("--volume_per_frame", type=float, default=2.25e-3)

    parser.add_argument("--add_ci", action="store_true", help="Add Poisson confidence intervals")

    args = parser.parse_args()

    # Expand ~
    args.sensor_csv = os.path.expanduser(args.sensor_csv)
    args.media_csv = os.path.expanduser(args.media_csv)

    main(args)
