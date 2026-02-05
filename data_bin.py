import os
import argparse
import pandas as pd
import pathlib
import logging
import numpy as np
from utils import identify_profiles  # JIT-accelerated profile detector

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# --------------------------------------------------
# CLI definition
# --------------------------------------------------
def cli():
    """
    Process a cruise's sensor + media data into per-profile depth-binned statistics.
    """
    parser = argparse.ArgumentParser(description="Process sensor and media data for a given cruise.")
    parser.add_argument('--cruise', type=str, required=True, help='Cruise ID, e.g. EN706')
    parser.add_argument('--sensor_dir', type=str, default='raw_data', help='Path to sensor CSVs')
    parser.add_argument('--media_dir', type=str, default='media_list/ISIIS1', help='Path to ISIIS1 CSVs')
    parser.add_argument('--media_dir2', type=str, default='media_list/ISIIS2', help='Path to ISIIS2 CSVs')
    parser.add_argument('--out_dir', type=str, default='dash_data/data/stingray', help='Output directory')
    parser.add_argument('--store_merge', action='store_true', help='Save merged raw data for debugging')
    parser.add_argument('--merged_dir', type=str, default='merged_data', help='Directory for merged raw CSV')
    parser.add_argument('--depth_bin', type=float, default=1.0, help='Depth bin size in meters')
    return parser.parse_args()


def main():
    args = cli()
    cruise = args.cruise
    sensor_dir = args.sensor_dir
    media_dir = args.media_dir
    media_dir2 = args.media_dir2
    out_dir = args.out_dir
    merged_dir = args.merged_dir
    depth_bin = args.depth_bin

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(merged_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) Load sensor CSV
    # --------------------------------------------------
    try:
        logging.info(f"Looking for sensor files in {sensor_dir}")
        sensor_file = next(
            f for f in os.listdir(sensor_dir)
            if f.endswith(".csv") and cruise in f
        )
    except StopIteration:
        logging.error(f"No sensor CSV found for cruise {cruise}")
        return

    logging.info(f"Reading sensor file: {sensor_file}")
    sensor_df = pd.read_csv(os.path.join(sensor_dir, sensor_file))
    sensor_df["times"] = pd.to_datetime(sensor_df["times"], errors="coerce")
    sensor_df = sensor_df.dropna(subset=["times"]).sort_values("times")

    # --------------------------------------------------
    # 2) Merge ISIIS1 (optional)
    # --------------------------------------------------
    try:
        media_file = next(
            f for f in os.listdir(media_dir)
            if f.endswith(".csv") and cruise in f
        )
        logging.info(f"Merging ISIIS1 file: {media_file}")
        media_df = pd.read_csv(os.path.join(media_dir, media_file))
        media_df["times"] = pd.to_datetime(media_df["times"], errors="coerce")
        media_df = media_df.dropna(subset=["times"]).sort_values("times")
        media_df = media_df.drop(columns=["id"], errors="ignore")

        df = pd.merge(sensor_df, media_df, on="times", how="outer").sort_values("times")

    except StopIteration:
        logging.warning("No ISIIS1 media found. Proceeding with sensor data only.")
        df = sensor_df.copy()

    # --------------------------------------------------
    # 3) Merge ISIIS2 (optional)
    # --------------------------------------------------
    try:
        media_file2 = next(
            f for f in os.listdir(media_dir2)
            if f.endswith(".csv") and cruise in f
        )
        logging.info(f"Merging ISIIS2 file: {media_file2}")
        media_df2 = pd.read_csv(os.path.join(media_dir2, media_file2))
        media_df2["times"] = pd.to_datetime(media_df2["times"], errors="coerce")
        media_df2 = media_df2.dropna(subset=["times"]).sort_values("times")
        media_df2 = media_df2.drop(columns=["id"], errors="ignore")

        df = pd.merge(df, media_df2, on="times", how="outer", suffixes=("", "_2")).sort_values("times")

    except StopIteration:
        logging.warning("No ISIIS2 media found. Proceeding without it.")

    df = df.reset_index(drop=True)

    # --------------------------------------------------
    # 4) Optional: store merged raw
    # --------------------------------------------------
    if args.store_merge and not df.empty:
        date_string = df["times"].iloc[0].strftime("%Y%m%d")
        merged_path = f"{merged_dir}/{date_string}_{cruise}_merged_raw.csv"
        df.to_csv(merged_path, index=False)
        logging.info(f"Merged raw data written to {merged_path}")

    # --------------------------------------------------
    # 5) Interpolate navigation metadata only
    # --------------------------------------------------
    logging.info("Interpolating navigation / metadata in time...")
    df = df.set_index("times").sort_index()

    interp_cols = [c for c in ["depth", "latitude", "longitude"] if c in df.columns]
    if "depth" not in interp_cols:
        logging.error("No 'depth' column found in data. Cannot identify profiles.")
        return

    for col in interp_cols:
        df[col] = df[col].interpolate(method="time")

    df = df.reset_index()

    # --------------------------------------------------
    # 6) Identify profiles on raw depth/time
    # --------------------------------------------------
    logging.info("Identifying CTD profiles...")
    df["profile"] = identify_profiles(
        depth=df["depth"].to_numpy(dtype=np.float64),
        time_seconds=df["times"].astype("int64").to_numpy() / 1e9
    )

    # --------------------------------------------------
    # 7) Bin depth within each profile (replace raw depth)
    # --------------------------------------------------
    logging.info(f"Binning depth within each profile (bin size = {depth_bin} m)...")
    df["depth"] = (
        df.groupby("profile", group_keys=False)["depth"]
          .apply(lambda x: np.floor((x + depth_bin / 2) / depth_bin) * depth_bin)
    )

    # --------------------------------------------------
    # 8) Group by (profile, depth)
    # --------------------------------------------------
    df["group"] = pd.factorize(pd.MultiIndex.from_arrays([df["profile"], df["depth"]]))[0]

    # --------------------------------------------------
    # 9) Identify sensor vs media columns
    # --------------------------------------------------
    sensor_cols = [c for c in sensor_df.columns if c not in ["timestamp", "matdate", "times", "depth"]]
    media_cols = [c for c in df.columns if c not in sensor_cols + ["group", "profile", "depth", "times"]]

    # --------------------------------------------------
    # 10) Aggregate by (profile, depth)
    # --------------------------------------------------
    logging.info("Aggregating statistics by (profile, depth)...")
    df_mean = df.groupby("group").agg(
        profile=("profile", "first"),
        depth=("depth", "first"),
        times=("times", "first"),
        **{col: (col, "first") for col in media_cols},
        **{f"{col}_std": (col, "std") for col in sensor_cols},
        **{f"{col}": (col, "mean") for col in sensor_cols},
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 11) Save output
    # --------------------------------------------------
    if df_mean.empty:
        logging.error("No aggregated data produced. Nothing to write.")
        return

    df_mean = df_mean.sort_values(["profile", "depth"])
    date_string = df_mean["times"].iloc[0].strftime("%Y%m%d")
    output_file = f"{out_dir}/{date_string}_{cruise}.csv"

    df_mean.to_csv(output_file, index=False)
    logging.info(f"Final processed data written to {output_file}")


if __name__ == "__main__":
    main()