# pipelines/process_cruise.py

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.config.columns import SLED_COLUMNS
from stingray.io.csv import read_csv_parallel
from stingray.io.indexing import load_or_build_file_index, filter_file_index
from stingray.utils.temporal import convert_timestamp
from stingray.utils.gridding import assign_time_bins
from stingray.utils.spatial import convert_gps
from stingray.sensors.ctd import density_ies80
from stingray.sensors.fluorometer import calibrate_chlorophyll, calibrate_backscatter
from stingray.sensors.par import calibrate_par
from stingray.sensors.suna import calibrate_nitrate
from stingray.profiles.identify import identify_profiles

logger = logging.getLogger(__name__)


def merge_sensors(
    cruise: str,
    start: str,
    end: str,
    root: str | Path = "sensor_data",
    cal_year: str = "2021",
    time_bin_seconds: float = 5.0,
    out_dir: str | Path = "dash_data/data/stingray/",
    media_list_dirs: list[str] | None = None,
    overwrite_index: bool = False,
) -> Path | None:
    if media_list_dirs is None:
        media_list_dirs = ["media_list/ISIIS1", "media_list/ISIIS2"]

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    root = Path(root)
    out_dir = Path(out_dir)

    Path("indexes").mkdir(exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # LOAD RAW SENSORS
    # -------------------------
    sensor_names = ["CTD", "DVL", "Fluorometer", "GPS", "Oxygen", "PAR", "SUNA"]
    sensors = {}

    for name in sensor_names:
        logger.info("Loading %s...", name)

        idx = load_or_build_file_index(
            root / name,
            Path("indexes") / f"{name}_index.csv",
            overwrite=overwrite_index,
        )

        files = filter_file_index(idx, start_date, end_date)
        sensors[name.lower()] = read_csv_parallel(files)

        logger.info(
            "%s: %s files, %s rows",
            name,
            len(files),
            len(sensors[name.lower()]),
        )

    # -------------------------
    # DEFINE TIME GRID FROM CTD
    # -------------------------
    ctd = sensors["ctd"].copy()

    if ctd.empty:
        logger.error("[%s] CTD missing — skipping cruise.", cruise)
        return None

    t_ctd = ctd["Timestamp"].to_numpy(float)
    grid_start = np.floor(np.nanmin(t_ctd) / time_bin_seconds) * time_bin_seconds
    grid_end = np.ceil(np.nanmax(t_ctd) / time_bin_seconds) * time_bin_seconds

    # -------------------------
    # ATTACH time_bin TO ALL SENSORS
    # -------------------------
    for key, df in sensors.items():
        if df.empty:
            df["time_bin"] = np.nan
            continue

        sensors[key]["time_bin"] = assign_time_bins(
            np.asarray(df["Timestamp"], dtype=np.float64),
            time_bin_seconds,
            grid_start,
            grid_end,
        )

    # -------------------------
    # CTD BLOCK
    # -------------------------
    ctd = sensors["ctd"]

    ctd["density"] = density_ies80(
        ctd["Salinity"].to_numpy(float),
        ctd["Temperature"].to_numpy(float),
        (ctd["Pressure"] / 10.0).to_numpy(float),
    )

    ctd_agg = (
        ctd.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg(
            {
                "Temperature": "median",
                "Conductivity": "median",
                "Pressure": "median",
                "Salinity": "median",
                "Sound Velocity": "median",
                "density": "median",
                "Depth": "first",
            }
        )
    )

    logger.info("CTD bins: %s", len(ctd_agg))

    # -------------------------
    # GPS BLOCK
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

    gps_agg = gps_agg.set_index("time_bin").reindex(ctd_agg["time_bin"]).reset_index()

    logger.info(
        "GPS bins raw: %s / %s",
        gps_agg[["Latitude", "Longitude"]].notna().all(axis=1).sum(),
        len(gps_agg),
    )

    # -------------------------
    # DVL BLOCK
    # -------------------------
    dvl = sensors["dvl"]

    dvl_agg = (
        dvl.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg(
            {
                "anglePitch": "median",
                "angleRoll": "median",
                "angleHeading": "median",
                "distanceAltitude": "median",
            }
        )
    )

    logger.info("DVL bins: %s", len(dvl_agg))

    # -------------------------
    # FLUOROMETER BLOCK
    # -------------------------
    fluoro = sensors["fluorometer"]

    if not fluoro.empty:
        fluoro["Chlorophyll"] = calibrate_chlorophyll(
            fluoro["Chlorophyll"],
            year=cal_year,
        )
        fluoro["Backscattering"] = calibrate_backscatter(
            fluoro["Backscattering"],
            year=cal_year,
        )

    fluoro_agg = (
        fluoro.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg({"Chlorophyll": "median", "Backscattering": "median"})
    )

    logger.info("Fluoro bins: %s", len(fluoro_agg))

    # -------------------------
    # OXYGEN BLOCK
    # -------------------------
    oxygen = sensors["oxygen"]

    oxygen_agg = (
        oxygen.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg({"O2Concentration": "median", "AirSaturation": "median"})
    )

    logger.info("Oxygen bins: %s", len(oxygen_agg))

    # -------------------------
    # PAR BLOCK
    # -------------------------
    par = sensors["par"]

    if not par.empty:
        par["Raw PAR [V]"] = calibrate_par(
            par["Raw PAR [V]"],
            year=cal_year,
        )

    par_agg = (
        par.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg({"Raw PAR [V]": "median"})
    )

    logger.info("PAR bins: %s", len(par_agg))

    # -------------------------
    # SUNA BLOCK
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

        suna["NitrateConcentration[uM]"] = calibrate_nitrate(suna, cruise)

    suna_agg = (
        suna.sort_values("Timestamp")
        .groupby("time_bin", as_index=False)
        .agg({"NitrateConcentration[uM]": "median"})
    )

    logger.info("SUNA bins: %s", len(suna_agg))

    # -------------------------
    # CAST / PROFILE SEGMENTATION
    # -------------------------
    cast, deployment = identify_profiles(
        depth=ctd_agg["Depth"].to_numpy(np.float64),
        time_seconds=ctd_agg["time_bin"].to_numpy(np.float64),
        bin_size=1.0,
        dir_window=4,
        max_gap_sec=300.0,
        min_turn_depth=10.0,
    )

    ctd_agg["cast"] = pd.Series(cast).astype("Int64")
    ctd_agg["deployment"] = pd.Series(deployment).astype("Int64")

    logger.info("Detected casts: %s", ctd_agg["cast"].nunique())
    logger.info("Detected deployments: %s", ctd_agg["deployment"].nunique())

    # -------------------------
    # MEDIA BLOCK
    # -------------------------
    media_aggs = []

    for media_dir in media_list_dirs:
        if not os.path.isdir(media_dir):
            continue

        media_file = next(
            (
                f
                for f in os.listdir(media_dir)
                if cruise in f and f.endswith(".csv")
            ),
            None,
        )

        if not media_file:
            continue

        tag = os.path.basename(media_dir).lower()
        logger.info("Processing media: %s", tag)

        media = pd.read_csv(os.path.join(media_dir, media_file))
        media["times"] = pd.to_datetime(media["times"], errors="coerce")
        media = media.dropna(subset=["times"]).sort_values("times")

        origin = datetime(1904, 1, 1)
        media["timestamp"] = (media["times"] - origin).dt.total_seconds()

        media["time_bin"] = assign_time_bins(
            np.asarray(media["timestamp"], dtype=np.float64),
            time_bin_seconds,
            grid_start,
            grid_end,
        )

        media_agg = (
            media.sort_values("timestamp")
            .groupby("time_bin", as_index=False)
            .agg(
                {
                    c: "first"
                    for c in media.columns
                    if c not in ["time_bin", "times", "timestamp"]
                }
            )
        )

        media_aggs.append(media_agg)
        logger.info("%s bins: %s", tag, len(media_agg))

    # -------------------------
    # MERGE MASTER TABLE
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
    # POST-MERGE PROCESSING
    # -------------------------
    sled["timestamp"] = sled["time_bin"]
    sled["times"] = pd.to_datetime(convert_timestamp(sled["time_bin"])[0])
    sled = sled.drop(columns=["time_bin"], errors="ignore")
    sled = sled.sort_values("timestamp")

    sled.rename(columns=SLED_COLUMNS, inplace=True)

    # -------------------------
    # QC: GPS INTERPOLATION INSIDE DEPLOYMENTS ONLY
    # -------------------------
    gps_cols = ["latitude", "longitude"]

    if all(c in sled.columns for c in gps_cols) and "deployment" in sled.columns:
        tmp = sled.set_index("times")[gps_cols].copy()
        dep = sled.set_index("times")["deployment"]

        for dep_id in dep.dropna().unique():
            mask = dep == dep_id
            tmp.loc[mask, gps_cols] = (
                tmp.loc[mask, gps_cols]
                .interpolate(method="time", limit_area="inside")
            )

        sled = sled.set_index("times")
        sled.loc[tmp.index, gps_cols] = tmp[gps_cols]
        sled = sled.reset_index()

        logger.info("QC: interpolated GPS inside deployments only.")

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    out_path = out_dir / f"{start_date.strftime('%Y%m%d')}_{cruise}.csv"
    sled.to_csv(out_path, index=False)

    logger.info("Saved: %s", out_path)
    logger.info("Final rows: %s", len(sled))

    return out_path