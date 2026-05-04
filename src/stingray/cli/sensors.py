from __future__ import annotations

import argparse
import logging

from stingray.logging.setup import setup_logging
from stingray.sensors.merge import merge_sensors


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stingray CTD-binned sensor aggregation + media + casts"
    )

    parser.add_argument("--cruise", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--root", default="sensor_data")
    parser.add_argument("--cal-year", default="2021")
    parser.add_argument("--time-bin-seconds", type=float, default=5.0)
    parser.add_argument("--out-dir", default="dash_data/data/stingray")
    parser.add_argument(
        "--media-list-dirs",
        nargs="*",
        default=["media_list/ISIIS1", "media_list/ISIIS2"],
    )
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    setup_logging(
        log_dir="logs",
        name=__name__,
        level=getattr(logging, args.log_level),
    )

    merge_sensors(
        cruise=args.cruise,
        start=args.start,
        end=args.end,
        root=args.root,
        cal_year=args.cal_year,
        time_bin_seconds=args.time_bin_seconds,
        out_dir=args.out_dir,
        media_list_dirs=args.media_list_dirs,
        overwrite_index=args.overwrite_index,
    )

if __name__ == "__main__":
    main()