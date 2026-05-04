from __future__ import annotations

import numpy as np


CHLOROPHYLL_CALIBRATIONS = {
    "default_year": "2021",
    "units": "ug/L",
    "2019": {"scale_factor": 0.0073, "dark_count": 48},
    "2021": {"scale_factor": 0.0073, "dark_count": 51},
}

BACKSCATTER_CALIBRATIONS = {
    "default_year": "2021",
    "units": "1/m sr",
    "2019": {"scale_factor": 1.684e-06, "dark_count": 48},
    "2021": {"scale_factor": 1.861e-06, "dark_count": 46},
}


def _get_calibration_params(calibrations: dict, year: str | None = None) -> dict:
    year = str(year or calibrations["default_year"])
    if year not in calibrations:
        raise ValueError(f"No calibration available for year {year}")
    return calibrations[year]


def calibrate_chlorophyll(raw_value, year: str | None = None):
    """
    Convert raw chlorophyll counts to calibrated chlorophyll concentration.
    """
    params = _get_calibration_params(CHLOROPHYLL_CALIBRATIONS, year)
    raw = np.asarray(raw_value, dtype=np.float64)
    return params["scale_factor"] * (raw - params["dark_count"])


def calibrate_backscatter(raw_value, year: str | None = None):
    """
    Convert raw backscatter counts to calibrated backscatter.
    """
    params = _get_calibration_params(BACKSCATTER_CALIBRATIONS, year)
    raw = np.asarray(raw_value, dtype=np.float64)
    return params["scale_factor"] * (raw - params["dark_count"])