from __future__ import annotations

import numpy as np


PAR_CALIBRATIONS = {
    "default_year": "2021",
    "units": "umol photons m^-2 s^-1",
    "2019": {"a0": 0.983748497, "a1": 8.129573432e-01, "Im": 1.3589},
    "2021": {"a0": 0.942280491, "a1": 8.215386e-01, "Im": 1.3589},
}


def _get_par_params(year: str | None = None) -> dict:
    year = str(year or PAR_CALIBRATIONS["default_year"])
    if year not in PAR_CALIBRATIONS:
        raise ValueError(f"No calibration available for year {year}")
    return PAR_CALIBRATIONS[year]


def calibrate_par(raw_value, year: str | None = None):
    """
    Convert raw PAR counts to calibrated PAR.
    """
    params = _get_par_params(year)
    raw = np.asarray(raw_value, dtype=np.float64)
    return params["Im"] * 10 ** ((raw - params["a0"]) / params["a1"])