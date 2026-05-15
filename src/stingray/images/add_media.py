# pipelines/add_media_to_merged.py

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _nearest_sensor_timestamp(
    media_times: np.ndarray,
    sensor_times: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(sensor_times, media_times)

    idx = np.clip(idx, 1, len(sensor_times) - 1)

    left = sensor_times[idx - 1]
    right = sensor_times[idx]

    nearest_idx = np.where(
        np.abs(media_times - left) <= np.abs(media_times - right),
        idx - 1,
        idx,
    )

    return sensor_times[nearest_idx]


def add_media_to_merged(
    merged_csv: str | Path,
    cruise: str,
    media_list_dirs: list[str] | None = None,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    if media_list_dirs is None:
        media_list_dirs = ["media_list/ISIIS1", "media_list/ISIIS2"]

    merged_csv = Path(merged_csv)

    if out_path is None:
        out_path = merged_csv
    else:
        out_path = Path(out_path)

    sled = pd.read_csv(merged_csv)

    if "timestamp" not in sled.columns:
        raise ValueError("Merged sensor file must contain 'timestamp' column.")

    sled = sled.sort_values("timestamp").copy()

    sensor_times = sled["timestamp"].to_numpy(dtype=np.float64)

    if len(sensor_times) < 2:
        raise ValueError("Merged sensor file must contain at least two timestamps.")

    for media_i, media_dir in enumerate(media_list_dirs, start=1):
        if not os.path.isdir(media_dir):
            logger.warning("Media directory not found: %s", media_dir)
            continue

        media_file = next(
            (
                f
                for f in os.listdir(media_dir)
                if cruise in f and f.endswith(".csv")
            ),
            None,
        )

        if media_file is None:
            logger.warning("No media CSV found for cruise %s in %s", cruise, media_dir)
            continue

        tag = os.path.basename(media_dir).lower()
        media_path = Path(media_dir) / media_file

        logger.info("Processing media: %s | %s", tag, media_path)

        media = pd.read_csv(media_path)

        if "times" not in media.columns:
            logger.warning("Skipping %s because it has no 'times' column.", media_path)
            continue

        media["times"] = pd.to_datetime(media["times"], errors="coerce")

        media = (
            media
            .dropna(subset=["times"])
            .sort_values("times")
            .copy()
        )

        if media.empty:
            logger.warning("Skipping %s because it has no valid media times.", media_path)
            continue

        origin = datetime(1904, 1, 1)

        media["timestamp"] = (
            media["times"] - origin
        ).dt.total_seconds()

        media["sensor_timestamp"] = _nearest_sensor_timestamp(
            media["timestamp"].to_numpy(dtype=np.float64),
            sensor_times,
        )

        media_agg = (
            media
            .sort_values("timestamp")
            .groupby("sensor_timestamp", as_index=False)
            .agg(
                {
                    c: "first"
                    for c in media.columns
                    if c not in [
                        "times",
                        "timestamp",
                        "sensor_timestamp",
                    ]
                }
            )
        )

        merge_key = "sensor_timestamp"

        if media_i > 1:
            suffix = f"_{media_i}"
            media_agg = media_agg.rename(
                columns={
                    c: f"{c}{suffix}"
                    for c in media_agg.columns
                    if c != merge_key
                }
            )

        media_cols = [
            c for c in media_agg.columns
            if c != merge_key
        ]

        existing_cols = [
            c for c in media_cols
            if c in sled.columns
        ]

        existing_with_data = [
            c for c in existing_cols
            if sled[c].notna().any()
        ]

        if existing_with_data and not overwrite:
            logger.info(
                "Skipping %s because media columns already contain data: %s",
                tag,
                existing_with_data,
            )
            continue

        if existing_cols and overwrite:
            logger.info(
                "Dropping existing media columns for %s: %s",
                tag,
                existing_cols,
            )
            sled = sled.drop(columns=existing_cols)

        sled = sled.merge(
            media_agg,
            left_on="timestamp",
            right_on=merge_key,
            how="left",
        )

        sled = sled.drop(columns=[merge_key], errors="ignore")

        logger.info(
            "%s media rows mapped to %s sensor timestamps.",
            tag,
            len(media_agg),
        )

    sled = sled.sort_values("timestamp")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sled.to_csv(out_path, index=False)

    logger.info("Saved media-enriched file to: %s", out_path)

    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Attach media-list fields to an already merged Stingray sensor CSV."
    )

    parser.add_argument("merged_csv")

    parser.add_argument("--cruise", required=True)

    parser.add_argument(
        "--media-list-dirs",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--out-path",
        default=None,
        help="Output CSV path. Defaults to updating merged_csv in place.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing populated media columns if present.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    add_media_to_merged(
        merged_csv=args.merged_csv,
        cruise=args.cruise,
        media_list_dirs=args.media_list_dirs,
        out_path=args.out_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()