#!/usr/bin/env python3
import os
import sys
import argparse
import pathlib
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from numba import njit
from utils import *

# =========================
# CLI
# =========================
def cli():
    p = argparse.ArgumentParser(description="CTD-defined time binning + sensor aggregation")
    p.add_argument("--cruise", required=True, help="Cruise ID (e.g., EN706)")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--root", default="/mnt/vast/nes-lter/Stingray/data/sensor_data", help="Root dir containing CTD/, DVL/, GPS/, etc.")
    p.add_argument("--cal-year", default="2021", help="Calibration year")
    p.add_argument("--time-bin-seconds", type=float, default=5.0, help="Time bin width in seconds")
    p.add_argument("--agg", choices=["mean", "median"], default="mean", help="Aggregation per bin")
    p.add_argument("--overwrite-index", action="store_true", help="Rebuild index CSVs")
    p.add_argument("--out-dir", default="raw_data", help="Output directory")
    p.add_argument("--log-dir", default="logs", help="Log directory")
    return p.parse_args()

# =========================
# MAIN
# =========================
def main():
    args = cli()
    setup_logging(args.log_dir)

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.end, "%Y-%m-%d")

    logging.info(f"User window: {start_date} .. {end_date}")

    pathlib.Path("indexes").mkdir(exist_ok=True)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sensor_names = ["CTD", "DVL", "Fluorometer", "GPS", "Oxygen", "PAR", "SUNA"]
    sensors = {}

    for name in sensor_names:
        sensor_dir = f"{args.root}/{name}"
        index_file = f"indexes/{name}_index.csv"

        file_index = load_or_build_file_index(sensor_dir, index_file, overwrite=args.overwrite_index)
        files = filter_file_index(file_index, start_date, end_date)
        df = read_csv_parallel(files)

        sensors[name.lower()] = df
        logging.info(f"{name}: files={len(files)} rows={len(df)}")

    # ---- CTD backbone ----
    ctd = sensors["ctd"]
    if ctd.empty:
        raise RuntimeError("CTD data missing – cannot define time grid")

    ctd = ctd.copy()
    ctd["Density"] = ies80(
        ctd["Salinity"].to_numpy(float),
        ctd["Temperature"].to_numpy(float),
        (ctd["Pressure"] / 10).to_numpy(float)
    )

    # Restore original timestamp conversion
    ctd["Times"], ctd["matdate"] = convert_timestamp(ctd["Timestamp"])

    t_ctd = ctd["Timestamp"].to_numpy(float)
    grid_start = np.floor(np.nanmin(t_ctd) / args.time_bin_seconds) * args.time_bin_seconds
    grid_end   = np.ceil(np.nanmax(t_ctd) / args.time_bin_seconds) * args.time_bin_seconds

    ctd = add_time_bin(ctd, "Timestamp", grid_start, grid_end, args.time_bin_seconds)
    assert not np.isnan(ctd["time_bin"]).any()

    # ---- Project all sensors ----
    for k in ["gps", "dvl", "fluorometer", "oxygen", "par", "suna"]:
        sensors[k] = add_time_bin(sensors[k], "Timestamp", grid_start, grid_end, args.time_bin_seconds)
        warn_unmatched(k.upper(), sensors[k])

    # ---- GPS conversion (RESTORED) ----
    if not sensors["gps"].empty:
        gps = sensors["gps"].copy()
        gps["Latitude"] = convert_gps(gps["Latitude"], gps["Latitude Hemisphere"])
        gps["Longitude"] = convert_gps(gps["Longitude"], gps["Longitude Hemisphere"])
        sensors["gps"] = gps
        logging.info("Converted GPS to decimal degrees")

    # ---- Fluorometer + PAR calibration (RESTORED) ----
    if not sensors["fluorometer"].empty:
        sensors["fluorometer"]["Chlorophyll"] = calibrate("chlorophyll", sensors["fluorometer"]["Chlorophyll"], args.cal_year)
        sensors["fluorometer"]["Backscattering"] = calibrate("backscatter", sensors["fluorometer"]["Backscattering"], args.cal_year)

    if not sensors["par"].empty:
        sensors["par"]["Raw PAR [V]"] = calibrate("par", sensors["par"]["Raw PAR [V]"], args.cal_year)
        
    # ---- Interpolate CTD T/S/P onto SUNA timestamps (for nitrate correction) ----
    if not sensors["suna"].empty:
        suna = sensors["suna"].copy()

        # Convert CTD timestamp to pandas datetime for interpolation
        ctd_interp = ctd[["Timestamp", "Salinity", "Temperature", "Pressure", "matdate"]].copy()
        ctd_interp["Times"] = pd.to_datetime(convert_timestamp(ctd_interp["Timestamp"])[0])
        ctd_interp = ctd_interp.sort_values("Times").set_index("Times")

        # SUNA times
        suna["Times"] = pd.to_datetime(convert_timestamp(suna["Timestamp"])[0])
        suna = suna.sort_values("Times").set_index("Times")

        # Time interpolation
        suna[["Salinity", "Temperature", "Pressure", "matdate"]] = (
            ctd_interp[["Salinity", "Temperature", "Pressure", "matdate"]]
            .reindex(suna.index, method="nearest")   # nearest is safest for SUNA spectra
        )

        suna = suna.reset_index(drop=True)
        sensors["suna"] = suna

        logging.info("Interpolated CTD T/S/P onto SUNA timestamps for nitrate correction")

    # ---- SUNA nitrate correction (RESTORED, pre-aggregation) ----
    if not sensors["suna"].empty:
        suna = sensors["suna"].copy()

        try:
            suna["NitrateConcentration[uM]"] = calibrate_nitrate(suna, args.cruise)
            sensors["suna"] = suna
            logging.info("Applied SUNA nitrate correction using interpolated CTD T/S/P")
        except Exception as e:
            logging.error("SUNA nitrate calibration failed – keeping raw values", exc_info=True)


    # ---- Aggregate ----
    agg = args.agg

    def agg_df(df, cols):
        if df.empty:
            return df
        return df.groupby("time_bin")[cols].agg(agg).reset_index()

    ctd_b = ctd.groupby("time_bin").agg(agg).reset_index()
    gps_b = agg_df(sensors["gps"], ["Latitude", "Longitude"])
    dvl_b = agg_df(sensors["dvl"], ["anglePitch", "angleRoll", "angleHeading", "distanceAltitude"])
    fluoro_b = agg_df(sensors["fluorometer"], ["Chlorophyll", "Backscattering"])
    oxy_b = agg_df(sensors["oxygen"], ["O2Concentration", "AirSaturation"])
    par_b = agg_df(sensors["par"], ["Raw PAR [V]"])
    suna_b = agg_df(sensors["suna"], ["NitrateConcentration[uM]"])

    sled = (
        ctd_b
        .merge(gps_b, on="time_bin", how="left")
        .merge(dvl_b, on="time_bin", how="left")
        .merge(fluoro_b, on="time_bin", how="left")
        .merge(par_b, on="time_bin", how="left")
        .merge(oxy_b, on="time_bin", how="left")
        .merge(suna_b, on="time_bin", how="left")
    )

    # ---- Final time columns ----
    sled["Timestamp"] = sled["time_bin"]
    sled["Times"] = pd.to_datetime(convert_timestamp(sled["time_bin"])[0])
    sled = sled.drop(columns=["time_bin"], errors="ignore")

    sled = sled.rename(columns=sled_columns)

    out_path = f"{args.out_dir}/{start_date.strftime('%Y%m%d')}_{args.cruise}.csv"
    sled.to_csv(out_path, index=False)

    logging.info(f"Saved: {out_path}")
    logging.info(f"Final rows: {len(sled)}")

if __name__ == "__main__":
    main()
