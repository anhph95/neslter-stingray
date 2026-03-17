#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
API_URL = "https://nes-lter-api.whoi.edu"
DEFAULT_MAX_WORKERS = os.cpu_count() - 1
def setup_logging(log_dir="logs", name="ctd_batch", level=logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
def cli():
    p = argparse.ArgumentParser(
        description="Download NES-LTER CTD cruise data, merge missing lat/lon/date from metadata when needed, and save one CSV per cruise."
    )
    p.add_argument("--out-dir", default="dash_data/data/ctd", help="Output directory for cruise CSVs")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Thread pool size for cast downloads")
    p.add_argument("--skip-existing", action="store_true", help="Skip cruise if any *_CRUISE.csv already exists")
    p.add_argument("--only-cruise", nargs="*", default=None, help="Optional list of cruise names to process")
    p.add_argument("--log-dir", default="logs", help="Directory for log files")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()
def cruise_file_exists(out_dir: Path, cruise: str) -> bool:
    return any(out_dir.glob(f"*_{cruise}.csv"))
def get_cast_list(cruise: str):
    try:
        metadata = pd.read_csv(
            f"{API_URL}/api/ctd/metadata/{cruise}",
            usecols=["cast", "latitude", "longitude", "date"],
        )
        cast_ids = metadata["cast"].dropna().astype(str).unique()
        return cast_ids, metadata
    except Exception:
        cast_list = pd.read_json(f"{API_URL}/api/ctd/casts/{cruise}")
        cast_ids = cast_list["number"].dropna().astype(str).unique()
        return cast_ids, None
def load_full_cruise_fast(cruise: str, max_workers: int, logger: logging.Logger) -> pd.DataFrame:
    cast_ids, metadata = get_cast_list(cruise)
    base = f"{API_URL}/api/ctd/cast/{cruise}/"
    failed_casts = []
    def fetch(cast_id: str):
        try:
            return pd.read_csv(base + str(cast_id))
        except Exception as e:
            failed_casts.append((cast_id, str(e)))
            return None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        dfs = list(ex.map(fetch, cast_ids))
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        raise RuntimeError(f"All casts failed for {cruise}")
    df = pd.concat(dfs, ignore_index=True)
    if metadata is not None:
        meta = metadata.copy()
        meta["cast"] = meta["cast"].astype(str)
        if "cast" in df.columns:
            df["cast"] = df["cast"].astype(str)
            merge_cols = []
            for col in ["latitude", "longitude", "date"]:
                if col not in df.columns:
                    merge_cols.append(col)
                elif df[col].isna().all():
                    merge_cols.append(col)
            if merge_cols:
                df = df.merge(meta[["cast"] + merge_cols], on="cast", how="left", suffixes=("", "_meta"))
                for col in merge_cols:
                    meta_col = f"{col}_meta"
                    if col in df.columns and meta_col in df.columns:
                        df[col] = df[col].where(df[col].notna(), df[meta_col])
                        df.drop(columns=[meta_col], inplace=True)
                    elif meta_col in df.columns:
                        df.rename(columns={meta_col: col}, inplace=True)
    if failed_casts:
        logger.warning(f"[{cruise}] {len(failed_casts)} casts failed")
        for cast_id, err in failed_casts[:10]:
            logger.warning(f"[{cruise}] cast {cast_id}: {err}")
    return df
def get_date_from_df(df: pd.DataFrame) -> str:
    if "date" not in df.columns:
        return "nodate"
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.notna().any():
        return dates.min().strftime("%Y%m%d")
    return "nodate"
def main():
    args = cli()
    logger = setup_logging(
        log_dir=args.log_dir,
        name="nes_lter_ctd_batch",
        level=getattr(logging, args.log_level),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading cruise list...")
    cruise_list = pd.read_csv(f"{API_URL}/api/ctd/cruises/all")
    if args.only_cruise:
        wanted = set(args.only_cruise)
        cruise_list = cruise_list[cruise_list["name"].isin(wanted)].copy()
        logger.info(f"Restricted to {len(cruise_list)} cruises: {sorted(wanted)}")
    total = len(cruise_list)
    logger.info(f"Total cruises to consider: {total}")
    for i, row in cruise_list.iterrows():
        cruise = row["name"]
        if args.skip_existing and cruise_file_exists(out_dir, cruise):
            logger.info(f"[{i+1}/{total}] Skipping {cruise} (file already exists)")
            continue
        try:
            logger.info(f"[{i+1}/{total}] Processing {cruise}...")
            df = load_full_cruise_fast(cruise, max_workers=args.max_workers, logger=logger)
            date_str = get_date_from_df(df)
            out_path = out_dir / f"{date_str}_{cruise}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"[{cruise}] Saved: {out_path}")
            logger.info(f"[{cruise}] Rows: {len(df)}")
        except Exception as e:
            logger.error(f"[{cruise}] Failed: {e}")
    logger.info("Done.")
if __name__ == "__main__":
    main()