### Import libraries
import os
import re
import glob
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.io import loadmat
import logging
from numba import njit
from pathlib import Path


# column_mappings.py
sled_columns = {
    'Timestamp': 'timestamp',
    'Times': 'times',
    'matdate': 'matdate',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'anglePitch': 'pitch',
    'angleRoll': 'roll',
    'angleHeading': 'heading',
    'distanceAltitude': 'altitude',
    'Temperature': 'temperature',
    'Conductivity': 'conductivity',
    'Pressure': 'pressure',
    'Depth': 'depth',
    'Salinity': 'salinity',
    'Sound Velocity': 'sound_velocity',
    'Density': 'density',
    'Density_IES80': 'density_ies80',
    'Chlorophyll': 'chlorophyll',
    'Backscattering': 'backscattering',
    'Raw PAR [V]': 'par',
    'O2Concentration': 'oxygen_concentration',
    'AirSaturation': 'oxygen_saturation',
    'NitrateConcentration[uM]': 'nitrate',
}

sled_units = {
    'latitude': '°',
    'longitude': '°',
    'pitch': '°',
    'roll': '°',
    'heading': '°',
    'altitude': 'm',
    'temperature': '°C',
    'conductivity': 'S m⁻¹',
    'pressure': 'dbar',
    'depth': 'm',
    'salinity': 'psu',
    'sound_velocity': 'm s⁻¹',
    'density': 'kg m⁻³',
    'density_ies80': 'kg m⁻³',
    'chlorophyll': 'µg l⁻¹',
    'backscattering': 'm⁻¹ sr⁻¹',
    'par': 'µmol photons m⁻² s⁻¹',
    'oxygen_concentration': 'µM',
    'oxygen_saturation': '%',
    'nitrate': 'µM',
}

calibration_data = {
    "chlorophyll": {
        "units": "µg/L",
        "default_year": "2021",
        "2019": {"params": {"scale_factor": 0.0073, "dark_count": 48}},
        "2021": {"params": {"scale_factor": 0.0073, "dark_count": 51}},
        "function": lambda raw, scale_factor, dark_count: scale_factor * (raw - dark_count)
    },
    "backscatter": {
        "units": "1/m sr",
        "default_year": "2021",
        "2019": {"params": {"scale_factor": 1.684e-06, "dark_count": 48}},
        "2021": {"params": {"scale_factor": 1.861e-06, "dark_count": 46}},
        "function": lambda raw, scale_factor, dark_count: scale_factor * (raw - dark_count)
    },
    "par": {
        "units": "µmol photons m⁻² s⁻¹",
        "default_year": "2021",
        "2019": {"params": {"a0": 0.983748497, "a1": 8.129573432e-01, "Im": 1.3589}},
        "2021": {"params": {"a0": 0.942280491, "a1": 8.215386e-01, "Im": 1.3589}},
        "function": lambda raw, a0, a1, Im: Im * 10 ** ((raw - a0) / a1)
    },
}

# Stingray sensor processing functions
def calibrate(sensor_type, raw_value, year=None, sign=None):
    """
    Apply the calibration function to raw sensor data.

    :param sensor_type: str, type of sensor ("chlorophyll", "backscatter", "par", "gps").
    :param raw_value: float, raw sensor reading.
    :param year: str, calibration year (required for non-GPS sensors; defaults to latest if not provided).
    :param sign: str, "N", "S", "E", "W" for GPS.
    :return: float, calibrated value.
    """  

    if sensor_type not in calibration_data:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    sensor_info = calibration_data[sensor_type]
    func = sensor_info["function"]

    year = year or sensor_info.get("default_year")
    if year not in sensor_info:
        raise ValueError(f"No calibration available for {sensor_type} in year {year}")

    params = sensor_info[year]["params"]

    raw = np.asarray(raw_value, dtype=np.float64)
    return func(raw, **params)

## SUNA data processing functions
# Function to parse MATLAB .mat file to python dictionary
def parse_mat(mat_path, name=None):
    """
    Parse a MATLAB .mat struct into a Python dictionary.
    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file.
    name : str, optional
        Variable name inside the .mat file. If omitted, uses the file stem.
    Returns
    -------
    dict
        Parsed MATLAB struct fields as a Python dictionary.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    if name is None:
        name = mat_path.stem
    mat = loadmat(mat_path)
    if name not in mat:
        valid_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Variable '{name}' not found in {mat_path}. "
            f"Available variables: {valid_keys}"
        )
    matfile = mat[name]
    data = {}
    if not hasattr(matfile, "dtype") or matfile.dtype.names is None:
        raise ValueError(f"Variable '{name}' in {mat_path} is not a MATLAB struct.")
    for field_name in matfile.dtype.names:
        data[field_name] = matfile[field_name][0][0]
    return data

# Function to check if required lines are in CAL path, needed for correction function
def check_lines(file_path, initials="", out_dir="modified_CAL"):
    """
    Ensure required metadata lines exist in a SUNA CAL file.
    If required lines are missing, write a modified CAL file to `out_dir`
    and return that new path. If the original file already contains the
    required lines, return the original file path.
    Parameters
    ----------
    file_path : str or Path
        Path to the input CAL file.
    initials : str, optional
        Suffix tag for the modified file name.
    out_dir : str or Path, optional
        Directory where modified CAL files should be written.
    Returns
    -------
    str
        Path to the usable CAL file (original or modified).
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_path = out_dir / f"{file_path.stem}_{initials}_modified.CAL"
    if out_path.exists():
        logging.info("Modified CAL file already exists.")
        return str(out_path)
    required_lines = """H,Pixel base,1,,,
H,Sensor Depth offset,0,,,
H,Br wavelength offset,210,,,
H,Min fit wavelength,,,,
H,Max fit wavelength,,,,
H,Use seawater dark current,No,,,
H,Pressure coef,0.0265,,,"""
    with open(file_path, "r") as file:
        content = file.read()
    if required_lines in content:
        logging.info("Required lines are already present. Using original CAL file.")
        return str(file_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    insert_index = content.rfind("H,File creation time")
    if insert_index == -1:
        logging.warning(
            f"Could not find insertion point in CAL file: {file_path}. "
            f"Using original file."
        )
        return str(file_path)
    content = content[:insert_index] + required_lines + "\n" + content[insert_index:]
    with open(out_path, "w") as file:
        file.write(content)
    logging.info("Required lines not present in original file.")
    logging.info(f"Generated modified CAL file: {out_path}")
    return str(out_path)

# Function to parse Nitrate calibration file
def parse_SOSIK_NO3cal(cal_file):
    cal = {
        "type": "SUNA",
        "SN": "",
        "CalTemp": None,
        "CalSDN": None,
        "CalDateStr": "",
        "DC_flag": 1,
        "pixel_base": None,
        "depth_lag": None,
        "WL_offset": None,
        "min_fit_WL": None,
        "max_fit_WL": None,
        "pres_coef": None,
    }
    replacements = {
        "New ": "",
        "DI DC Corr": "Ref",
        "Reference": "Ref",
        "Wavelength": "WL",
        "WaveLen": "WL",
        ",NO3": ",ENO3",
        ",SWA": ",ESW",
        ",ASW,": ",ESW,",
        ",T*ASW": ",TSWA",
    }
    hdr = None
    try:
        with open(cal_file, "r") as f:
            lines = f.readlines()
        data_lines = []
        for line in lines:
            tline = line.strip().split(",")
            if not tline or len(tline) < 2:
                continue
            if tline[0] == "H":
                field = tline[1]
                if field.startswith("SUNA"):
                    cal["type"] = "SUNA"
                    m = re.search(r"SUNA\s+(\d+)", field)
                    if m:
                        cal["SN"] = m.group(1)
                if field.startswith("Lamp#"):
                    cal["SN"] = " ".join(tline[1:])
                m = re.search(r"\d+/\d+/\d{4}", field)
                if m:
                    d_str = m.group(0)
                    cal["CalDateStr"] = d_str
                    cal["CalSDN"] = datetime.strptime(d_str, "%m/%d/%Y").date()
                if "creation time" in field:
                    m = re.search(r"\d{2}-\w{3}-\d{4}", field)
                    if m:
                        d_str = m.group(0)
                        cal["CalDateStr"] = d_str
                        cal["CalSDN"] = datetime.strptime(d_str, "%d-%b-%Y").date()
                if "CalTemp" in field:
                    cal["CalTemp"] = float(tline[2]) if len(tline) > 2 and tline[2] else None
                if re.search(r"T_CAL", field) and cal["CalTemp"] is None:
                    parts = field.split(" ")
                    if len(parts) > 1:
                        try:
                            cal["CalTemp"] = float(parts[1])
                        except ValueError:
                            pass
                if "Pixel base" in field:
                    cal["pixel_base"] = int(tline[2]) if len(tline) > 2 and tline[2] else None
                if "Sensor Depth" in field:
                    vals = tline[2:4]
                    cal["depth_lag"] = [float(x) if x else None for x in vals]
                if "Br wavelength" in field:
                    cal["WL_offset"] = float(tline[2]) if len(tline) > 2 and tline[2] else None
                if "Min fit" in field:
                    cal["min_fit_WL"] = float(tline[3]) if len(tline) > 3 and tline[3] else None
                if "Max fit" in field:
                    cal["max_fit_WL"] = float(tline[3]) if len(tline) > 3 and tline[3] else None
                if "Use seawater dark" in field:
                    cal["DC_flag"] = 0 if "yes" in field.lower() else 1
                if "Pressure coef" in field:
                    cal["pres_coef"] = float(tline[2]) if len(tline) > 2 and tline[2] else None
                if "Wavelength" in tline and "NO3" in tline:
                    hdr_line = ",".join(tline)
                    for old, new in replacements.items():
                        hdr_line = hdr_line.replace(old, new)
                    hdr = hdr_line.split(",")[1:]
            elif tline[0] == "E":
                data_lines.append(tline[1:])
        if len(data_lines) == 0:
            logging.info(f"File exists but no data lines found for: {cal_file}")
            return cal
        if hdr is None:
            logging.warning(f"No wavelength/data header found in calibration file: {cal_file}")
            return cal
        data = np.array(data_lines, dtype=float)
        if data.shape[1] != len(hdr):
            logging.warning(
                f"Header/data width mismatch in {cal_file}: "
                f"{data.shape[1]} data columns vs {len(hdr)} header columns"
            )
            n = min(data.shape[1], len(hdr))
            data = data[:, :n]
            hdr = hdr[:n]
        for i, name in enumerate(hdr):
            cal[name] = data[:, i]
    except FileNotFoundError:
        logging.info(f"Cannot find calibration file at: {cal_file}")
        return cal
    return cal


# Bofu's correction algorithm
def calc_bofu_no3(spec, ncal, pcor_flag):
    """
    Apply SUNA nitrate correction algorithm.
    Parameters
    ----------
    spec : dict
        Spectral data dictionary.
    ncal : dict
        Parsed nitrate calibration dictionary.
    pcor_flag : int
        Whether to apply pressure correction (1=yes, 0=no).
    Returns
    -------
    dict
        Output dictionary containing corrected nitrate and fit diagnostics.
    """
    WL_offset = ncal["WL_offset"]
    d = spec.copy()
    d["P"] = d["STP"][2]
    d["T"] = d["STP"][1]
    d["S"] = d["STP"][0]
    del d["STP"]
    if ncal["pixel_base"] == 0:
        d["spectra_pix_range"] += 1
        d["pix_fit_win"] += 1
        logging.info("Pixel registration offset by +1")
    DC = d["DC"] if ncal["DC_flag"] != 0 else d["SWDC"]
    if ncal["min_fit_WL"]:
        d["pix_fit_win"][0] = np.where(ncal["WL"] >= ncal["min_fit_WL"])[0][0]
        d["WL_fit_win"][0] = ncal["WL"][d["pix_fit_win"][0]]
    if ncal["max_fit_WL"]:
        d["pix_fit_win"][1] = np.where(ncal["WL"] <= ncal["max_fit_WL"])[0][-1]
        d["WL_fit_win"][1] = ncal["WL"][d["pix_fit_win"][1]]
    if d["spectra_WL_range"][0][0] > d["WL_fit_win"][0][0]:
        logging.info(
            f"Min WL of returned spectra ({d['spectra_WL_range'][0][0]}) is greater "
            f"than Min WL of fit window ({d['WL_fit_win'][0][0]})."
        )
        d["WL_fit_win"][0][0] = d["spectra_WL_range"][0][0]
        logging.info(
            f"Fit window adjusted [{d['WL_fit_win'][0][0]} {d['WL_fit_win'][0][1]}] "
            "and NO3 estimate may be compromised."
        )
    if d["spectra_WL_range"][0][1] < d["WL_fit_win"][0][1]:
        logging.info(
            f"Max WL of returned spectra ({d['spectra_WL_range'][0][1]}) is less "
            f"than Max WL of fit window ({d['WL_fit_win'][0][1]})."
        )
        d["WL_fit_win"][0][1] = d["spectra_WL_range"][0][1]
        logging.info(
            f"Fit window adjusted [{d['WL_fit_win'][0][0]} {d['WL_fit_win'][0][1]}] "
            "and NO3 estimate may be compromised."
        )
    if np.isnan(np.sum(d["pix_fit_win"])) and "WL_fit_win" in d:
        pfit_low = np.where(ncal["WL"] >= d["WL_fit_win"][0][0])[0][0]
        pfit_hi = np.where(ncal["WL"] <= d["WL_fit_win"][0][1])[0][-1]
        d["pix_fit_win"] = [pfit_low, pfit_hi]
    ind1 = np.where(ncal["WL"] >= d["spectra_WL_range"][0][0])[0][0]
    ind2 = np.where(ncal["WL"] <= d["spectra_WL_range"][0][1])[0][-1]
    d["spectra_pix_range"] = [ind1, ind2]
    saved_pixels = np.arange(d["spectra_pix_range"][0], d["spectra_pix_range"][1] + 1)
    rows, cols = d["UV_INTEN"].shape
    REF = np.tile(ncal["Ref"][saved_pixels], (rows, 1))
    WL = np.tile(ncal["WL"][saved_pixels], (rows, 1))
    ESW = np.tile(ncal["ESW"][saved_pixels], (rows, 1))
    ENO3 = ncal["ENO3"][saved_pixels]
    UV_INTEN_SW = d["UV_INTEN"] - DC
    if np.count_nonzero(UV_INTEN_SW <= 0):
        UV_INTEN_SW[UV_INTEN_SW <= 0] = np.nan
        logging.info("UV intensities <= dark current found. Setting to NaN.")
    ratio = UV_INTEN_SW / REF
    bad_ratio = (~np.isfinite(ratio)) | (ratio <= 0)
    if np.any(bad_ratio):
        ratio[bad_ratio] = np.nan
        logging.info("Non-positive or invalid UV/REF ratios found. Setting to NaN.")
    ABS_SW = -np.log10(ratio)
    ctd_temp = np.tile(d["T"], (cols, 1)).T
    ctd_sal = np.tile(d["S"], (cols, 1)).T
    cal_temp = np.tile(ncal["CalTemp"], (rows, cols))
    Tcorr_coef = [1.27353e-07, -7.56395e-06, 2.91898e-05, 1.67660e-03, 1.46380e-02]
    Tcorr = np.polyval(Tcorr_coef, (WL - WL_offset)) * (ctd_temp - cal_temp)
    ESW_in_situ = ESW * np.exp(Tcorr)
    if pcor_flag == 1:
        pres_term = (1 - d["P"] / 1000 * ncal["pres_coef"]).reshape(-1, 1)
        ESW_in_situ *= pres_term
    ABS_Br_tcor = ESW_in_situ * ctd_sal
    ABS_cor = ABS_SW - ABS_Br_tcor
    t_fit = (saved_pixels >= d["pix_fit_win"][0]) & (saved_pixels <= d["pix_fit_win"][1])
    Fit_WL = WL[0, t_fit]
    Fit_ENO3 = ENO3[t_fit]
    Ones = np.ones_like(Fit_ENO3)
    M = np.column_stack((Fit_ENO3, Ones / 100, Fit_WL / 1000))
    M_INV = np.linalg.pinv(M)
    colsM = M.shape[1]
    NO3 = np.full((rows, colsM + 3), np.nan)
    for i in range(rows):
        samp_ABS_cor = ABS_cor[i, t_fit]
        if np.all(np.isnan(samp_ABS_cor)):
            continue
        NO3[i, :3] = M_INV @ samp_ABS_cor
        NO3[i, 1] /= 100
        NO3[i, 2] /= 1000
        ABS_BL = WL[i, :] * NO3[i, 2] + NO3[i, 1]
        ABS_NO3_EXP = ENO3 * NO3[i, 0]
        FIT_DIF = ABS_cor[i, :] - ABS_BL - ABS_NO3_EXP
        RMS_ERROR = np.sqrt(np.nansum(FIT_DIF[t_fit] ** 2) / np.sum(t_fit))
        ind_240 = np.where(Fit_WL <= 240)[0][-1]
        ABS_240 = [Fit_WL[ind_240], samp_ABS_cor[ind_240]]
        NO3[i, colsM:colsM + 3] = [RMS_ERROR, *ABS_240]
    out = {
        "hdr": [
            "SDN",
            "AVG DC",
            "Pres",
            "Temp",
            "Sal",
            "SBE_NO3, uM/L",
            "NO3, uM/L",
            "BL_B",
            "BL_M",
            "RMS ERROR",
            "WL~240",
            "ABS~240",
        ],
        "info": {
            "WL_fit_window": d["WL_fit_win"],
            "spectra_WL_range": d["spectra_WL_range"],
            "WL_offset": WL_offset,
        },
    }
    avg_DC = np.nanmean(DC, axis=1)
    NO3 = np.column_stack((d["SDN"], avg_DC, d["P"], d["T"], d["S"], d["NO3"], NO3))
    tnan = np.isnan(NO3[:, 5])
    out["data"] = NO3[~tnan, :]
    return out

def calibrate_nitrate(df, CRUISE, nitrate_col="NitrateConcentration[uM]", dark_col="DarkValueUsedForFit"):
    """
    Calibrate nitrate concentration from SUNA sensor data.
    This version preserves alignment with the original dataframe rows and
    does not mutate the caller's dataframe in place.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing SUNA data.
    CRUISE : str
        Cruise identifier used to find the calibration file.
    nitrate_col : str, optional
        Raw nitrate column name.
    dark_col : str, optional
        Dark-current column name.
    Returns
    -------
    np.ndarray
        Corrected nitrate concentrations aligned to the original input rows.
    """
    df = df.copy()
    out_full = np.full(len(df), np.nan, dtype=float)
    df[nitrate_col] = pd.to_numeric(df[nitrate_col], errors="coerce")
    if dark_col in df.columns:
        df[dark_col] = pd.to_numeric(df[dark_col], errors="coerce")
    pkg_dir = Path(__file__).resolve().parent
    data_ref_dir = pkg_dir / "data_reference"
    spec_path = data_ref_dir / "spec.mat"
    hdr_file = data_ref_dir / "suna_hdr.txt"
    cal_dir = Path("suna_calibration")
    try:
        cal_file = next(file for file in os.listdir(cal_dir) if CRUISE.upper() in file)
        cal_path = cal_dir / cal_file
    except StopIteration:
        logging.info(f"No calibration file found for cruise {CRUISE}, using raw nitrate values.")
        raw = df[nitrate_col].to_numpy(dtype=float)
        if dark_col in df.columns:
            raw[df[dark_col].to_numpy(dtype=float) == 0] = np.nan
        return raw
    cal_path = Path(check_lines(str(cal_path), "AP"))
    spec = parse_mat(spec_path)
    ncal = parse_SOSIK_NO3cal(str(cal_path))
    with open(hdr_file, "r") as file:
        nitrate_hdr = file.readline().strip().split(",")
    uv_cols = [col for col in df.columns if re.search(r"SpectrumCh", col)]
    uv_list = [col for col in nitrate_hdr if re.search(r"UV\(\d+(\.\d+)?\)", col)]
    if len(uv_cols) != len(uv_list):
        raise ValueError(
            f"Mismatch between dataframe spectrum columns ({len(uv_cols)}) "
            f"and SUNA header UV columns ({len(uv_list)})."
        )
    rename_map = dict(zip(uv_cols, uv_list))
    df = df.rename(columns=rename_map)
    required_cols = [nitrate_col, "Salinity", "Temperature", "Pressure", "matdate", *uv_list]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for nitrate calibration: {missing}")
    df[uv_list] = df[uv_list].apply(pd.to_numeric, errors="coerce")
    valid = df[nitrate_col].notna()
    if not valid.any():
        return out_full
    work = df.loc[valid].copy()
    work_idx = work.index.to_numpy()
    spec["STP"] = [
        work["Salinity"].to_numpy(dtype=float),
        work["Temperature"].to_numpy(dtype=float),
        work["Pressure"].to_numpy(dtype=float),
    ]
    spec["UV_INTEN"] = work[uv_list].to_numpy(dtype=float)
    spec["DC"] = np.outer(work[dark_col].to_numpy(dtype=float), np.ones(len(uv_list)))
    spec["NO3"] = work[nitrate_col].to_numpy(dtype=float)
    spec["SDN"] = work["matdate"].to_numpy()
    spec["spectra_WL_range"][0][0] = ncal["WL"][0]
    spec["spectra_WL_range"][0][1] = ncal["WL"][-1]
    result = calc_bofu_no3(spec, ncal, pcor_flag=1)
    corrected = result["data"][:, 6]
    if len(corrected) != len(work_idx):
        logging.warning(
            f"Corrected nitrate length ({len(corrected)}) does not match valid input rows "
            f"({len(work_idx)}). Attempting positional assignment to available rows only."
        )
        n = min(len(corrected), len(work_idx))
        out_full[work_idx[:n]] = corrected[:n]
    else:
        out_full[work_idx] = corrected
    if dark_col in work.columns:
        dark_zero = work[dark_col].to_numpy(dtype=float) == 0
        out_full[work_idx[dark_zero]] = np.nan
    logging.info(f"Calibration file found for cruise {CRUISE}, TSP correction applied.")
    return out_full

# Convert GPS
def convert_gps(raw_series, sign_series):
    signs = sign_series.map({'N': 1, 'S': -1, 'E': 1, 'W': -1})
    return signs * (np.floor(raw_series / 100) + (raw_series % 100) / 60)

### Date processing functions

# Function to convert datetime object(s) to MATLAB serial date numbers
def datenum(dts):
    # Define the base date (MATLAB uses Jan 0, 0000, but Python uses a different base)
    base_date = datetime(1, 1, 1)
    # Function to calculate the serial date number for a single datetime object
    def date2num(dt):
        if isinstance(dt, pd.Timestamp):
            dt = pd.to_datetime(dt).round("us").to_pydatetime()
        elif not isinstance(dt, datetime):
            raise TypeError("Input must be a datetime object or pandas.Timestamp")
        delta = dt - base_date
        return delta.days + delta.seconds / 86400 + 366 + 1
    # Check if the input is a list of datetime objects
    if isinstance(dts, (list, np.ndarray, pd.Series)):
        # Process each datetime object in the list
        return [date2num(dt) for dt in dts]
    elif isinstance(dts, (datetime, pd.Timestamp)):
        # Process a single datetime object
        return date2num(dts)
    else:
        raise TypeError("Input must be a datetime object or a list of datetime objects")

# Function to convert serial date number(s)/timestamp(s) back to datetime object
def datestr(date_numbers, fmt=0, as_str=False):
    # MATLAB base date is 0000-01-01, using closest Python date 0001-01-01
    # Adjust by one day because Python's datetime starts at 0001-01-01
    matlab_base_date = datetime(1, 1, 1)
    # Define format options similar to MATLAB's `datestr`
    formats = {
        0: '%d-%b-%Y %H:%M:%S',  # Default format
        1: '%d-%b-%Y',            # Day-Month-Year
        2: '%m/%d/%y',            # Month/Day/Year
        3: '%Y-%m-%d',            # Year-Month-Day
        4: '%d/%m/%Y',            # Day/Month/Year
        5: '%H:%M:%S',            # Time
    }
    def num2date(date_number):
        # Convert serial date number to datetime
        date_time_obj = matlab_base_date + timedelta(days=date_number - 366.0 -1.0)
        if as_str:
            # Use custom format if provided
            if isinstance(fmt, str):
                return date_time_obj.strftime(fmt)
            else:
                return date_time_obj.strftime(formats.get(fmt, '%d-%b-%Y %H:%M:%S'))
        else:
            return date_time_obj
    # Check if the input is a single serial date number or a list
    if isinstance(date_numbers, (int, float)):  # Single int or float
        return num2date(date_numbers)
    elif isinstance(date_numbers, (list, pd.Series, np.ndarray)):  # List, Series, or ndarray
        return [num2date(num) for num in date_numbers]
    elif date_numbers is None:  # Handle None case
        return None
    else:
        raise ValueError("Input should be an int, float, list, pandas Series, or numpy ndarray.")

# Function to convert timestamp to datetime and MATLAB serial date number
def convert_timestamp(timestamp, origin_date = datetime(1904, 1, 1)):
    times = pd.to_datetime(timestamp, unit="s", origin=origin_date)
    matdate = datenum(times)
    return times, matdate

def rawdate2date(series):
    # Extract year and day from the series and convert it to datetime in one step
    year = series.astype(str).str[:4]
    day_of_year = series.astype(str).str[4:].astype(int) - 1
    return pd.to_datetime(year, format='%Y') + pd.to_timedelta(day_of_year, unit='D')

def rawtime2time(series):
    # Convert string to float, divide by 24 to get a fraction of a day, then convert to timedelta
    day_fractions = series.astype(float) / 24
    return pd.to_timedelta(day_fractions, unit='D').dt.components.apply(
        lambda x: f"{x['hours']:02}:{x['minutes']:02}:{x['seconds']:02}", axis=1
    )
    
def matdate2timestamp(matdate, origin_date=datetime(1904, 1, 1)):
    def one(x):
        dt = datestr(x, as_str=False)  # MATLAB datenum -> datetime
        return (dt - origin_date).total_seconds()
    if isinstance(matdate, (int, float)):
        return one(matdate)
    elif isinstance(matdate, (list, np.ndarray, pd.Series)):
        return [one(x) for x in matdate]
    else:
        raise TypeError("Input must be scalar or array-like")

### Data processing functions
## New sensor data processing functions
# Function to read multiple CSV files concurrently with exception handling directly in the parallel process
def read_csv_parallel(file_list, max_workers=None):
    """
    Reads multiple CSV files in parallel using ThreadPoolExecutor and concatenates them into a single DataFrame.
    Exceptions are handled directly in the parallel process.
    """
    import pandas as pd
    import os, concurrent.futures
    def safe_read_csv(file):
        try:
            return pd.read_csv(file, low_memory=False)
        except Exception as e:
            logging.info(f"Failed to read {file}: {e}")
            return pd.DataFrame()  # Empty if failed
    # Pick sensible number of workers
    if max_workers is None:
        max_workers = max(os.cpu_count() - 1, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(safe_read_csv, file_list))
    # ⚠️ Filter out empty DataFrames to avoid deprecated concat([]) call
    dfs = [df for df in dfs if not df.empty]
    if dfs:  # only concat if we actually have data
        concatenated_df = pd.concat(dfs, ignore_index=True)
    else:
        concatenated_df = pd.DataFrame()
    return concatenated_df

def build_file_index(dir_root):
    """
    Scan directory and return a list of (datetime, filepath) pairs.
    """
    import glob, os
    from datetime import datetime
    list_of_pairs = []
    list_all = glob.glob(f"{dir_root}/*.csv")
    for file in list_all:
        try:
            basename = os.path.basename(file).rsplit(".", 1)[0]
            date_str = basename.split("-", 1)[1]  # after first dash
            file_date = datetime.strptime(date_str, "%Y-%m-%d %H-%M-%S.%f")
            list_of_pairs.append((file_date, file))
        except Exception as e:
            logging.info(f"⚠️ Skipped {file}: {e}")
    return list_of_pairs

def filter_file_index(file_index_df, start_date, end_date):
    """Filter file index DataFrame for files within a given date range (inclusive)."""
    if "datetime" not in file_index_df.columns:
        raise ValueError("file_index_df must contain a 'datetime' column")
    if end_date is None:
        end_date = datetime.max
    else:
        end_date = end_date + timedelta(days=1)  # make inclusive
    mask = (file_index_df["datetime"] >= start_date) & (file_index_df["datetime"] < end_date)
    return file_index_df.loc[mask, "filepath"].tolist()


def save_file_index(dir_root, out_file):
    """Build and save index as CSV with standardized column names."""
    index = build_file_index(dir_root)
    df = pd.DataFrame(index, columns=["datetime", "filepath"])
    df.to_csv(out_file, index=False)
    return df


def load_or_build_file_index(dir_root, out_file, overwrite=False):
    """
    Load an existing CSV index if available, or build a new one.
    """
    if os.path.exists(out_file) and not overwrite:
        logging.info(f"Using cached index: {out_file}")
        df = pd.read_csv(out_file, parse_dates=["datetime"])
    else:
        if overwrite:
            logging.info(f"Rebuilding index (overwrite requested): {out_file}")
        else:
            logging.info(f"No cached index found. Building new index: {out_file}")
        df = save_file_index(dir_root, out_file)
    return df


# Function to merge two dataframes using merge_asof and optionally mask duplicates
def merge_df(df1, df2, on, cols=None, direction='backward', duplicates=False):
    """
    Merges two dataframes using merge_asof and optionally masks duplicates in specified columns.
    Parameters:
    - df1 (pd.DataFrame): The first dataframe.
    - df2 (pd.DataFrame): The second dataframe.
    - on (str): The column to merge on.
    - cols (list): List of columns to merge from df2. If None, uses columns from df2 that don't overlap with df1.
    - direction (str): Direction for merge_asof. Default is 'backward'.
    - duplicates (bool): If False, masks duplicate matches in df2.
    Returns:
    - pd.DataFrame: The merged dataframe, optionally with duplicates masked.
    """    
    # Make a copy of the dataframes to avoid modifying the original data
    df1 = df1.copy().sort_values(on).reset_index(drop=True)
    df2 = df2.copy().sort_values(on).reset_index(drop=True)
    # If cols is None, get non-overlapping columns from df2
    if not cols:
        cols = df2.columns.difference(df1.columns).tolist()
    # Add an 'id' column to track original rows in df2
    df2['id'] = df2.index
    # Perform the merge using merge_asof
    merged_df = pd.merge_asof(df1, df2[[on] + cols + ['id']], on=on, direction=direction)
    # Mask duplicates if the duplicates flag is False
    if not duplicates:
        merged_df[cols] = merged_df[cols].mask(merged_df.duplicated(subset='id'))
    # Drop the 'id' column as it's no longer needed
    merged_df.drop(columns='id', inplace=True)
    return merged_df


## Data binning
# Function to bin data based on specified columns and steps
import numpy as np
import pandas as pd

def bin_data(df, cols, steps):
    """
    Bin numeric or datetime columns by given step sizes.
    NaT values are preserved (not dropped).
    """
    df = df.copy()
    for col, step in zip(cols, steps):
        if np.issubdtype(df[col].dtype, np.datetime64):
            # Initialize with NaT
            binned = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            # Only apply binning to valid datetimes
            valid = df[col].notna()
            binned.loc[valid] = pd.to_datetime(
                ((df.loc[valid, col].astype("int64") + step // 2) // step) * step
            )
            df[f"{col}_bin"] = binned
        else:
            df[f"{col}_bin"] = np.floor((df[col] + step / 2) / step) * step
    return df

def bin_vectors(vectors, steps):
    """
    Bin numeric or datetime vectors by given step sizes, with simple hysteresis.
    For each input vector in `vectors` and its corresponding step in `steps`:
    1. We first compute a "naive" bin using standard quantization:
           numeric:   floor((x + step/2) / step) * step
           datetime:  round to nearest bin in ns using the same idea on int64
    2. Then we apply a 1D hysteresis rule along the original element order:
           - We keep track of:
               last_raw : the last *committed* raw value
               last_bin : the bin assigned to that committed value
           - For each new sample raw[i]:
               if |raw[i] - last_raw| < step:
                   treat as noise  → assign bin[i] = last_bin
               else:
                   treat as real move → accept bin[i] as computed,
                                        and update last_raw, last_bin
       This means:
         - Tiny fluctuations less than `step` around the last committed value
           will NOT cause a bin change.
         - A change of at least `step` (or more) in the raw value will create
           a new bin.
         - This works for any numeric vector and for datetime64 vectors (in ns).
    3. NaNs / NaT:
         - Datetime NaT values are preserved and never binned.
         - Numeric NaNs keep their quantized value, but they do NOT update
           hysteresis state (they are effectively skipped in the hysteresis
           comparison).
    The function returns a list of binned vectors, one per input vector.
    """
    """
    Prepare a list to collect the binned output vectors.
    """
    outputs = []
    """
    Loop over each input vector and its corresponding step size.
    """
    for vec, step in zip(vectors, steps):
        # Convert to NumPy array (do not copy unless needed)
        raw = np.asarray(vec)
        # ------------------------------------------------------------------ #
        # Datetime64 case
        # ------------------------------------------------------------------ #
        if np.issubdtype(raw.dtype, np.datetime64):
            """
            For datetime vectors:
            1. Work only on valid (non-NaT) entries.
            2. Convert datetimes to int64 nanoseconds for arithmetic.
            3. Quantize (round) using the same midpoint rule:
                   q = round_to_nearest_step(raw_ns, step_ns)
            4. Apply hysteresis on the raw_ns values:
                   compare current raw_ns to last committed raw_ns.
            """
            # Initialize output array with NaT so missing values are preserved
            out = np.full(raw.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            # Boolean mask for non-NaT values
            valid = ~np.isnat(raw)
            # Raw datetime values as int64 nanoseconds (only where valid)
            raw_ns = raw[valid].astype("int64")
            """
            Step 1: naive binning (quantization).
            We do midpoint rounding by adding step//2 before integer division:
              - (raw_ns + step//2) // step   gives the bin index
              - * step                      rescales to ns units
            """
            q_ns = ((raw_ns + step // 2) // step) * step
            """
            Step 2: hysteresis on actual sample difference.
            We walk through raw_ns in original order and maintain:
              last_raw : last committed raw ns value
              last_bin : bin assigned at that commitment
            For each index i:
                if this is the first valid sample:
                    commit it directly (seed hysteresis)
                else:
                    if |raw_ns[i] - last_raw| < step:
                        → noise → override q_ns[i] with last_bin
                    else:
                        → real move → accept q_ns[i] and update last_raw, last_bin
            """
            last_raw = None
            last_bin = None
            for i in range(raw_ns.size):
                current_raw = raw_ns[i]
                if last_raw is None:
                    # First valid sample: seed hysteresis state
                    last_raw = current_raw
                    last_bin = q_ns[i]
                else:
                    # Compare actual change in ns to step
                    if abs(current_raw - last_raw) < step:
                        # Too small: treat as noise, keep previous bin
                        q_ns[i] = last_bin
                    else:
                        # Big enough: commit a new bin and update hysteresis state
                        last_raw = current_raw
                        last_bin = q_ns[i]
            # Convert back to datetime64[ns] and fill the output array
            out[valid] = q_ns.astype("datetime64[ns]")
            outputs.append(out)
        # ------------------------------------------------------------------ #
        # Numeric case
        # ------------------------------------------------------------------ #
        else:
            """
            For numeric vectors:
            1. We compute the naive quantized bin:
                   q = floor((x + step/2) / step) * step
               which is midpoint rounding to the nearest step.
            2. We apply the same hysteresis idea as for datetimes, but now
               using raw numeric values instead of ns.
            """
            raw = raw.astype(float, copy=False)
            """
            Step 1: naive numeric binning.
            - Add step/2 before division to implement midpoint rounding.
            - floor(...) then multiply by step gives the quantized bin center.
            """
            q = np.floor((raw + step / 2.0) / step) * step
            """
            Step 2: hysteresis on actual sample difference.
            We iterate in element order and keep:
              last_raw : last committed raw numeric value
              last_bin : bin assigned at that commitment
            For each index i:
                - If raw[i] is NaN:
                    * skip it for hysteresis (do not update last_raw/last_bin)
                    * leave q[i] as is (NaN remains NaN)
                - Else:
                    * If this is the first non-NaN:
                          seed hysteresis with this raw/bin.
                    * Else:
                          if |raw[i] - last_raw| < step:
                              → noise → set q[i] = last_bin
                          else:
                              → real move → accept q[i] and update
                                            last_raw, last_bin
            """
            last_raw = None
            last_bin = None
            for i in range(raw.size):
                current_raw = raw[i]
                # Treat NaN as a gap: do not update hysteresis state
                if np.isnan(current_raw):
                    continue
                if last_raw is None:
                    # First non-NaN: seed hysteresis
                    last_raw = current_raw
                    last_bin = q[i]
                else:
                    if abs(current_raw - last_raw) < step:
                        # Small change → noise → stay in the same bin
                        q[i] = last_bin
                    else:
                        # Large enough → commit new bin and update hysteresis
                        last_raw = current_raw
                        last_bin = q[i]
            outputs.append(q)
    """
    Return the list of binned vectors, one per input vector.
    """
    return outputs

# Function got get ranges from midpoints
def mid2range(midpoints):
    # Initialize the ranges list with the first range boundary
    ranges = [midpoints[0] - (midpoints[1] - midpoints[0]) / 2]
    
    # Calculate midpoints between each adjacent pair
    for i in range(len(midpoints) - 1):
        midpoint = (midpoints[i] + midpoints[i + 1]) / 2
        ranges.append(midpoint)
        
    # Append the last range boundary
    ranges.append(midpoints[-1] + (midpoints[-1] - midpoints[-2]) / 2)
    
    return np.array(ranges)

# Function to read a single file
def read_file(filename, hdr=None):
    try:
        if hdr is True:
            data_ref_dir = Path(__file__).resolve().parent / "data_reference"
            hdr_path = data_ref_dir / "suna_hdr.txt"
            with open(hdr_path, "r") as f:
                hdr = f.readline().strip().split(",")
        temp = pd.read_csv(filename, skiprows=14, header=None)
        lhs = pd.DataFrame({
            0: pd.Series([float('nan')] * len(temp), dtype='float64'),
            1: rawdate2date(temp.iloc[:, 1]),
            2: rawtime2time(temp.iloc[:, 2])
        })
        rhs = temp.iloc[:, 3:-6].copy()
        df = pd.concat([lhs, rhs], axis=1)
        if hdr:
            df.columns = hdr[:-8]
        return df
    except Exception as e:
        logging.info(f"Failed to read {filename}: {e}")
        return pd.DataFrame()
    
## Get nearest stations
class StationLocator:
    DEFAULT_STATION_REF_URL = "https://nes-lter-api.whoi.edu/api/stations/file"
    def __init__(
        self,
        station_reference: pd.DataFrame | str | None = None,
        max_distance_km: float = 2.0
    ):
        if station_reference is None:
            st = pd.read_csv(self.DEFAULT_STATION_REF_URL)
        elif isinstance(station_reference, str):
            st = pd.read_csv(station_reference)
        else:
            st = station_reference.copy()
        st.columns = st.columns.str.strip()
        required = ['station', 'startDate', 'endDate', 'decimalLatitude', 'decimalLongitude']
        missing = [c for c in required if c not in st.columns]
        if missing:
            raise ValueError(f"station_reference missing required columns: {missing}")
        st['station'] = st['station'].astype(str).str.strip()
        st['startDate'] = pd.to_datetime(st['startDate'], errors='coerce', utc=True).dt.tz_localize(None)
        st['endDate'] = st['endDate'].replace('current', pd.NA)
        st['endDate'] = pd.to_datetime(st['endDate'], errors='coerce', utc=True).dt.tz_localize(None)
        st['endDate_filled'] = st['endDate'].fillna(pd.Timestamp('2100-12-31'))
        st['decimalLatitude'] = pd.to_numeric(st['decimalLatitude'], errors='coerce')
        st['decimalLongitude'] = pd.to_numeric(st['decimalLongitude'], errors='coerce')
        st = st.dropna(subset=['station', 'startDate', 'decimalLatitude', 'decimalLongitude']).copy()
        self.station_reference = st
        self.max_distance_km = max_distance_km
    def active_stations(self, timestamp) -> pd.DataFrame:
        ts = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.isna(ts):
            raise ValueError("timestamp is missing or invalid")
        ts = ts.tz_localize(None)
        active = self.station_reference[
            (self.station_reference['startDate'] <= ts) &
            (ts <= self.station_reference['endDate_filled'])
        ].copy()
        if active.empty:
            raise ValueError(f"No active stations found for timestamp {ts}")
        return active
    def station_distances(self, lat, lon, timestamp) -> pd.Series:
        active = self.active_stations(timestamp)
        distances = []
        index = []
        for row in active.itertuples():
            d_km = geo_distance(
                (lat, lon),
                (row.decimalLatitude, row.decimalLongitude)
            ).km
            distances.append(d_km)
            index.append(row.Index)
        return pd.Series(distances, index=index)
    def nearest_station(self, lat, lon, timestamp, max_distance_km: float | None = None):
        distances = self.station_distances(lat, lon, timestamp)
        i = distances.idxmin()
        d_km = distances.loc[i]
        threshold = self.max_distance_km if max_distance_km is None else max_distance_km
        if threshold is not None and d_km > threshold:
            return np.nan, np.nan
        station_name = self.station_reference.loc[i, 'station']
        return station_name, d_km
    def nearest_stations(
        self,
        df,
        lat_col='latitude',
        lon_col='longitude',
        time_col='sample_time',
        max_distance_km: float | None = None
    ):
        names, distances = [], []
        for row in df.itertuples():
            lat = getattr(row, lat_col)
            lon = getattr(row, lon_col)
            ts = getattr(row, time_col)
            if pd.isna(lat) or pd.isna(lon) or pd.isna(ts):
                names.append(np.nan)
                distances.append(np.nan)
                continue
            name, dist = self.nearest_station(lat, lon, ts, max_distance_km=max_distance_km)
            names.append(name)
            distances.append(dist)
        return names, distances

    
from numba import njit
import numpy as np
@njit
def ies80(s, t, p):
    n = s.shape[0]
    rho = np.empty(n, dtype=np.float64)
    for i in range(n):
        si = s[i]
        ti = t[i]
        pi = p[i]
        # r0 coefficients
        r0 = (
            999.842594
            + 6.793952e-2 * ti
            - 9.09529e-3 * ti**2
            + 1.001685e-4 * ti**3
            - 1.120083e-6 * ti**4
            + 6.536332e-9 * ti**5
            + (8.24493e-1 - 4.0899e-3 * ti + 7.6438e-5 * ti**2 - 8.2467e-7 * ti**3 + 5.3875e-9 * ti**4) * si
            + (-5.72466e-3 + 1.0227e-4 * ti - 1.6546e-6 * ti**2) * si**1.5
            + 4.8314e-4 * si**2
        )
        if pi != 0.0:
            K = (
                19652.21
                + 148.4206 * ti
                - 2.327105 * ti**2
                + 1.360447e-2 * ti**3
                - 5.155288e-5 * ti**4
                + (3.239908 + 1.43713e-3 * ti + 1.16092e-4 * ti**2 - 5.77905e-7 * ti**3) * pi
                + (8.50935e-5 - 6.12293e-6 * ti + 5.2787e-8 * ti**2) * pi**2
                + (54.6746 - 0.603459 * ti + 1.09987e-2 * ti**2 - 6.1670e-5 * ti**3) * si
                + (7.944e-2 + 1.6483e-2 * ti - 5.3009e-4 * ti**2) * si**1.5
                + (2.2838e-3 - 1.0981e-5 * ti - 1.6078e-6 * ti**2) * pi * si
                + 1.91075e-4 * pi * si**1.5
                + (9.9348e-7 + 2.0816e-8 * ti + 9.1697e-10 * ti**2) * pi**2 * si
            )
            rho[i] = r0 / (1.0 - pi / K)
        else:
            rho[i] = r0
    return rho


### Data merge utils
import os, sys, logging
from datetime import datetime

def setup_logging(log_dir="logs", name=None, level=logging.INFO):
    """
    Universal logging setup:
      - logs/<script>_<timestamp>.out.log  (INFO+)
      - logs/<script>_<timestamp>.err.log  (ERROR+)
      - console output (INFO+)
    Log format:
      INFO | message
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)  # root if name=None
    logger.setLevel(level)
    fmt = logging.Formatter("%(levelname)s | %(message)s")
    # Prevent duplicate handlers if called multiple times in same process
    if logger.handlers:
        return logger
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = os.path.join(log_dir, f"{script}_{ts}.out.log")
    stderr_path = os.path.join(log_dir, f"{script}_{ts}.err.log")
    stdout_handler = logging.FileHandler(stdout_path)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)
    stderr_handler = logging.FileHandler(stderr_path)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.addHandler(console)
    logger.info(f"Logging to: {stdout_path}")
    logger.info(f"Errors to: {stderr_path}")
    return logger

# =========================
# NUMBA: TIME BINNING
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


def add_time_bin(df, ts_col, grid_start, grid_end, bin_width):
    if df.empty:
        return df
    df = df.copy()
    ts = np.asarray(df[ts_col], dtype=np.float64)
    df["time_bin"] = assign_time_bins(ts, bin_width, grid_start, grid_end)
    return df


def warn_unmatched(name, df):
    if df.empty:
        return
    n_bad = np.isnan(df["time_bin"]).sum()
    if n_bad > 0:
        logging.error(f"{name}: {n_bad} rows FAILED to map to CTD time bins")

from numba import njit
@njit
def identify_profiles(
    depth,
    time_seconds,
    bin_size=1,
    smooth_window=7,
    dir_window=10,
    max_gap_sec=3600.0,
    min_samples=10,
    min_depth_threshold=20.0,
    min_turn_depth=5.0,
    min_turn_time=60.0,
):
    """
    Identify CTD profiles (casts) from depth and time.
    This function segments a continuous depth time series into separate
    physical profiles (downcasts, upcasts, and separate deployments).
    The algorithm is designed to be robust to:
      - wire jitter and ship heave
      - short winch pauses
      - noisy depth measurements
      - small yo-yo motion near turning points
    It uses quantization + median smoothing BEFORE estimating direction,
    then detects real turnarounds with hysteresis, and enforces hard breaks
    on time gaps.
    Parameters
    ----------
    depth : 1D array of float
        Raw depth measurements (positive downward).
    time_seconds : 1D array of float
        Time in seconds (monotonic increasing).
    bin_size : float
        Depth bin size (meters) used for midpoint quantization.
        Acts like a deadband: depth changes smaller than this are ignored.
    smooth_window : int
        Window size (samples) for rolling median smoothing of quantized depth.
        Suppresses spikes and high-frequency jitter.
    dir_window : int
        Window size (samples) used to estimate direction of motion.
        Larger = smoother direction but more lag near turning points.
    max_gap_sec : float
        Time gap (seconds) that forces a new profile (e.g., recovery/redeploy).
    min_samples : int
        Minimum number of unique depth bins required for a profile to be valid.
        Short micro-profiles are treated as noise.
    min_depth_threshold : float
        Profiles that never get shallower than this depth are considered invalid.
        Prevents shallow dithering near surface from becoming real profiles.
    min_turn_depth : float
        Minimum depth reversal (meters) required to confirm a turnaround.
        Suppresses tiny yo-yo motion at the bottom.
    min_turn_time : float
        Minimum time (seconds) after the last extremum before confirming
        a turnaround. Prevents rapid oscillations from splitting profiles.
    Returns
    -------
    profile_out : 1D float array
        Profile labels for each sample.
        Labels are contiguous integers starting at 0.
        NaN indicates samples that belong to invalid profiles with no valid
        neighbor to merge into.
    deployment_id : 1D int array
        Deployment labels for each sample.
        Increments only across hard time gaps.
        Not affected by turnaround logic or merging.
    """
    n = depth.size
    # -----------------------------
    # Step 0: Midpoint quantization (deadband / hysteresis in depth space)
    # -----------------------------
    depth_q = np.empty(n)
    half_bin = bin_size / 2.0
    for i in range(n):
        d = depth[i]
        if np.isnan(d):
            depth_q[i] = np.nan
        else:
            depth_q[i] = np.floor((d + half_bin) / bin_size) * bin_size
    # -----------------------------
    # Step 1: Rolling median smoothing on quantized depth
    # -----------------------------
    smooth = np.empty(n)
    hw = smooth_window // 2
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        buf = []
        for j in range(lo, hi):
            v = depth_q[j]
            if not np.isnan(v):
                buf.append(v)
        if len(buf) == 0:
            smooth[i] = np.nan
        else:
            buf.sort()
            smooth[i] = buf[len(buf) // 2]
    # -----------------------------
    # Step 2: Direction estimate (upcast vs downcast)
    # -----------------------------
    direction = np.zeros(n)
    direction[:] = np.nan
    half = dir_window // 2
    for i in range(half, n - half):
        prev_med = smooth[i - half]
        next_med = smooth[i + half]
        if np.isnan(prev_med) or np.isnan(next_med):
            continue
        d = next_med - prev_med
        if d > 0:
            direction[i] = 1
        elif d < 0:
            direction[i] = -1
    # -----------------------------
    # Step 3: Profile boundaries
    #         (hard gaps + physical turnarounds)
    # -----------------------------
    profile = np.zeros(n, dtype=np.int64)
    hard_profile = np.zeros(n, dtype=np.bool_)
    # NEW: deployment id (hard gaps only)
    deployment_id = np.zeros(n, dtype=np.int64)
    current_deployment = 0
    deployment_id[0] = 0
    last_dir = 0.0
    last_extreme_depth = smooth[0]
    last_extreme_time = time_seconds[0]
    last_extreme_idx = 0
    for i in range(1, n):
        gap = (time_seconds[i] - time_seconds[i - 1]) > max_gap_sec
        if gap:
            # Increment deployment
            current_deployment += 1
            deployment_id[i] = current_deployment
            new_pid = profile[i - 1] + 1
            profile[i] = new_pid
            hard_profile[new_pid] = True
            last_dir = 0.0
            last_extreme_depth = smooth[i]
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
            continue
        # Same deployment
        deployment_id[i] = current_deployment
        profile[i] = profile[i - 1]
        d = direction[i]
        if np.isnan(d) or np.isnan(smooth[i]):
            continue
        dz = smooth[i] - last_extreme_depth
        dt = time_seconds[i] - last_extreme_time
        if last_dir == 0:
            last_dir = d
            last_extreme_depth = smooth[i]
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
            continue
        if d != last_dir and abs(dz) >= min_turn_depth and dt >= min_turn_time:
            new_pid = profile[i - 1] + 1
            for k in range(last_extreme_idx + 1, i + 1):
                profile[k] = new_pid
            last_dir = d
            last_extreme_depth = smooth[i]
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
        else:
            if (last_dir > 0 and smooth[i] > last_extreme_depth) or \
               (last_dir < 0 and smooth[i] < last_extreme_depth):
                last_extreme_depth = smooth[i]
                last_extreme_time = time_seconds[i]
                last_extreme_idx = i
    profile_out = profile.astype(np.float64)
    # -----------------------------
    # Step 4: Validate profiles + merge tiny noise segments
    # -----------------------------
    max_pid = profile[-1] + 1
    is_valid = np.zeros(max_pid, dtype=np.bool_)
    for pid in range(max_pid):
        seen = 0
        last = np.nan
        min_depth = 1e20
        for i in range(n):
            if profile[i] == pid:
                d = depth_q[i]
                if not np.isnan(d):
                    if d < min_depth:
                        min_depth = d
                    if np.isnan(last) or d != last:
                        seen += 1
                        last = d
        is_valid[pid] = (seen >= min_samples) and (min_depth <= min_depth_threshold)
    for pid in range(max_pid):
        if is_valid[pid] or hard_profile[pid]:
            continue
        prev_valid = -1
        for p in range(pid - 1, -1, -1):
            if is_valid[p]:
                prev_valid = p
                break
        if prev_valid >= 0:
            for i in range(n):
                if profile[i] == pid:
                    profile_out[i] = prev_valid
        else:
            for i in range(n):
                if profile[i] == pid:
                    profile_out[i] = np.nan
    # -----------------------------
    # Step 5: Relabel profiles contiguously
    # -----------------------------
    mapping = {}
    new_id = 0
    for i in range(n):
        pid = profile_out[i]
        if not np.isnan(pid) and pid not in mapping:
            mapping[pid] = new_id
            new_id += 1
    for i in range(n):
        pid = profile_out[i]
        if not np.isnan(pid):
            profile_out[i] = mapping[pid]
    return profile_out, deployment_id
