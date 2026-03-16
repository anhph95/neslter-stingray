#!/usr/bin/env python3
import os
import pathlib
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from utils import *
from numba import njit


# =========================
# CLI
# =========================
def cli():
    p = argparse.ArgumentParser(
        description="Stingray CTD-binned sensor aggregation + media + casts (dashboard build)"
    )
    p.add_argument("--cruise", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--root", default="sensor_data", help="Root sensor data dir")
    p.add_argument("--cal-year", default="2021")
    p.add_argument("--time-bin-seconds", type=float, default=5.0)
    p.add_argument("--out-dir", default="dash_data/data/stingray_timebinned/")
    p.add_argument("--media-list-dirs", nargs="*", default=["media_list/ISIIS1", "media_list/ISIIS2"])
    p.add_argument("--overwrite-index", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# =========================
# MAIN
# =========================
def main():
    args = cli()
    logger = setup_logging(log_dir="logs", name=__name__, level=getattr(logging, args.log_level))

    CRUISE = args.cruise
    START = args.start
    END = args.end
    ROOT = args.root
    CAL_YEAR = args.cal_year
    TIME_BIN_SECONDS = args.time_bin_seconds
    OUT_DIR = args.out_dir
    MEDIA_LIST_DIRS = args.media_list_dirs
    OVERWRITE_INDEX = args.overwrite_index

    start_date = datetime.strptime(START, "%Y-%m-%d")
    end_date = datetime.strptime(END, "%Y-%m-%d")

    pathlib.Path("indexes").mkdir(exist_ok=True)
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # LOAD RAW SENSORS
    # -------------------------
    sensor_names = ["CTD", "DVL", "Fluorometer", "GPS", "Oxygen", "PAR", "SUNA"]
    sensors = {}

    for name in sensor_names:
        logger.info(f"Loading {name}...")
        idx = load_or_build_file_index(
            f"{ROOT}/{name}",
            f"indexes/{name}_index.csv",
            overwrite=OVERWRITE_INDEX
        )
        files = filter_file_index(idx, start_date, end_date)
        sensors[name.lower()] = read_csv_parallel(files)
        logger.info(f"{name}: {len(files)} files, {len(sensors[name.lower()])} rows")

    # -------------------------
    # DEFINE TIME GRID (CTD)
    # -------------------------
    ctd = sensors["ctd"].copy()
    if ctd.empty:
        logger.error(f"[{CRUISE}] CTD missing — skipping cruise.")
        return

    t_ctd = ctd["Timestamp"].to_numpy(float)
    grid_start = np.floor(np.nanmin(t_ctd) / TIME_BIN_SECONDS) * TIME_BIN_SECONDS
    grid_end = np.ceil(np.nanmax(t_ctd) / TIME_BIN_SECONDS) * TIME_BIN_SECONDS

    # -------------------------
    # ATTACH time_bin TO ALL SENSORS
    # -------------------------
    for k in sensors:
        sensors[k]["time_bin"] = assign_time_bins(
            np.asarray(sensors[k]["Timestamp"], dtype=np.float64),
            TIME_BIN_SECONDS,
            grid_start,
            grid_end,
        )

    # -------------------------
    # CTD BLOCK
    # -------------------------
    ctd = sensors["ctd"]
    ctd["density"] = ies80(
        ctd["Salinity"].to_numpy(float),
        ctd["Temperature"].to_numpy(float),
        (ctd["Pressure"] / 10).to_numpy(float),
    )

    ctd_agg = (
        ctd.sort_values("Timestamp")
           .groupby("time_bin", as_index=False)
           .agg({
               "Temperature": "median",
               "Conductivity": "median",
               "Pressure": "median",
               "Salinity": "median",
               "Sound Velocity": "median",
               "density": "median",
               "Depth": "first",
           })
    )
    logger.info(f"CTD bins: {len(ctd_agg)}")

    # -------------------------
    # GPS BLOCK (RAW BINS ONLY — NO INTERP IN PIPELINE)
    # -------------------------
    gps = sensors["gps"]
    if not gps.empty:
        gps["Latitude"] = convert_gps(gps["Latitude"], gps["Latitude Hemisphere"])
        gps["Longitude"] = convert_gps(gps["Longitude"], gps["Longitude Hemisphere"])

    gps_agg = (
        gps.sort_values("Timestamp")
           .groupby("time_bin", as_index=False)
           .agg({"Latitude": "first", "Longitude": "first"})
    )

    # Align to CTD grid (preserve NaNs)
    gps_agg = gps_agg.set_index("time_bin").reindex(ctd_agg["time_bin"]).reset_index()

    logger.info(
        f"GPS bins (raw): {gps_agg[['Latitude','Longitude']].notna().all(axis=1).sum()} / {len(gps_agg)}"
    )

    # -------------------------
    # DVL BLOCK
    # -------------------------
    dvl = sensors["dvl"]
    dvl_agg = (
        dvl.sort_values("Timestamp")
           .groupby("time_bin", as_index=False)
           .agg({
               "anglePitch": "median",
               "angleRoll": "median",
               "angleHeading": "median",
               "distanceAltitude": "median",
           })
    )
    logger.info(f"DVL bins: {len(dvl_agg)}")

    # -------------------------
    # FLUOROMETER BLOCK
    # -------------------------
    fluoro = sensors["fluorometer"]
    if not fluoro.empty:
        fluoro["Chlorophyll"] = calibrate("chlorophyll", fluoro["Chlorophyll"], CAL_YEAR)
        fluoro["Backscattering"] = calibrate("backscatter", fluoro["Backscattering"], CAL_YEAR)

    fluoro_agg = (
        fluoro.sort_values("Timestamp")
              .groupby("time_bin", as_index=False)
              .agg({"Chlorophyll": "median", "Backscattering": "median"})
    )
    logger.info(f"Fluoro bins: {len(fluoro_agg)}")

    # -------------------------
    # OXYGEN BLOCK
    # -------------------------
    oxygen = sensors["oxygen"]
    oxygen_agg = (
        oxygen.sort_values("Timestamp")
              .groupby("time_bin", as_index=False)
              .agg({"O2Concentration": "median", "AirSaturation": "median"})
    )
    logger.info(f"Oxygen bins: {len(oxygen_agg)}")

    # -------------------------
    # PAR BLOCK
    # -------------------------
    par = sensors["par"]
    if not par.empty:
        par["Raw PAR [V]"] = calibrate("par", par["Raw PAR [V]"], CAL_YEAR)

    par_agg = (
        par.sort_values("Timestamp")
           .groupby("time_bin", as_index=False)
           .agg({"Raw PAR [V]": "median"})
    )
    logger.info(f"PAR bins: {len(par_agg)}")

    # -------------------------
    # SUNA BLOCK (CALIBRATE AT SUNA TIMES USING NEAREST CTD SUPPORT)
    # -------------------------
    suna = sensors["suna"]

    if not suna.empty:
        if "Times" not in ctd.columns:
            ctd["Times"], _ = convert_timestamp(ctd["Timestamp"])

        suna["Times"], suna["matdate"] = convert_timestamp(suna["Timestamp"])

        suna = suna.sort_values("Timestamp")
        ctd_sorted = ctd.sort_values("Timestamp")

        suna = pd.merge_asof(
            suna,
            ctd_sorted[["Timestamp", "Salinity", "Temperature", "Pressure"]],
            on="Timestamp",
            direction="nearest",
            allow_exact_matches=True,
        )

        suna["NitrateConcentration[uM]"] = calibrate_nitrate(suna, CRUISE)

    suna_agg = (
        suna.sort_values("Timestamp")
            .groupby("time_bin", as_index=False)
            .agg({"NitrateConcentration[uM]": "median"})
    )
    logger.info(f"SUNA bins: {len(suna_agg)}")

    # -------------------------
    # CAST (PROFILE SEGMENTATION ON CTD BINS)
    # -------------------------
    cast = identify_profiles(
        depth=ctd_agg["Depth"].to_numpy(np.float64),
        time_seconds=ctd_agg["time_bin"].to_numpy(np.float64),
        bin_size=1.0,
        dir_window=4,
        max_gap_sec=300.0,
        min_turn_depth=10.0,
    ).astype(np.int64)

    ctd_agg["cast"] = cast
    logger.info(f"Detected casts: {ctd_agg['cast'].nunique()}")

    # -------------------------
    # MEDIA BLOCK (MULTI-SYSTEM)
    # -------------------------
    media_aggs = []
    for media_dir in MEDIA_LIST_DIRS:
        if not os.path.isdir(media_dir):
            continue

        media_file = next(
            (f for f in os.listdir(media_dir) if CRUISE in f and f.endswith(".csv")),
            None,
        )
        if not media_file:
            continue

        tag = os.path.basename(media_dir).lower()
        logger.info(f"Processing media: {tag}")

        media = pd.read_csv(os.path.join(media_dir, media_file))
        media["times"] = pd.to_datetime(media["times"], errors="coerce")
        media = media.dropna(subset=["times"]).sort_values("times")

        ORIGIN = datetime(1904, 1, 1)
        media["timestamp"] = (media["times"] - ORIGIN).dt.total_seconds()

        media["time_bin"] = assign_time_bins(
            np.asarray(media["timestamp"], dtype=np.float64),
            TIME_BIN_SECONDS,
            grid_start,
            grid_end,
        )

        media_agg = (
            media.sort_values("timestamp")
                 .groupby("time_bin", as_index=False)
                 .agg({c: "first" for c in media.columns if c not in ["time_bin", "times", "timestamp"]})
        )

        media_aggs.append(media_agg)
        logger.info(f"{tag} bins: {len(media_agg)}")

    # -------------------------
    # MERGE (MASTER TABLE)
    # -------------------------
    sled = (
        ctd_agg
        .merge(gps_agg, on="time_bin", how="left")
        .merge(dvl_agg, on="time_bin", how="left")
        .merge(fluoro_agg, on="time_bin", how="left")
        .merge(par_agg, on="time_bin", how="left")
        .merge(oxygen_agg, on="time_bin", how="left")
        .merge(suna_agg, on="time_bin", how="left")
    )

    for i, media_agg in enumerate(media_aggs, start=1):
        if i == 1:
            sled = sled.merge(media_agg, on="time_bin", how="left")
        else:
            suffix = f"_{i}"
            cols = [c for c in media_agg.columns if c != "time_bin"]
            sled = sled.merge(
                media_agg.rename(columns={c: f"{c}{suffix}" for c in cols}),
                on="time_bin",
                how="left",
            )

    # -------------------------
    # Post-merge processing
    # -------------------------
    sled["timestamp"] = sled["time_bin"]
    sled["times"] = pd.to_datetime(convert_timestamp(sled["time_bin"])[0])
    sled = sled.drop(columns=["time_bin"], errors="ignore")

    sled = sled.sort_values("timestamp")

    # Rename columns to dashboard schema
    sled.rename(columns=sled_columns, inplace=True)

    # -------------------------
    # QC: TIME-INTERPOLATE GPS AFTER MERGE (BUILT-IN)
    # -------------------------
    # This is a QC tweak, not part of raw binning. Comment this block out if you want raw GPS only.
    sled[["latitude", "longitude"]] = (
        sled.set_index("times")[["latitude", "longitude"]]
            .interpolate(method="time", limit_area="inside")
            .reset_index(drop=True)
    )
    logger.info("QC: interpolated GPS by time (inside-only).")

    # ------------------------
    # Save output
    # ------------------------
    out_path = f"{OUT_DIR}/{start_date.strftime('%Y%m%d')}_{CRUISE}.csv"
    sled.to_csv(out_path, index=False)

    logger.info(f"Saved: {out_path}")
    logger.info(f"Final rows: {len(sled)}")


if __name__ == "__main__":
    main()
