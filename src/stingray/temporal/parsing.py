from __future__ import annotations

import pandas as pd


def rawdate2date(series: pd.Series) -> pd.Series:
    """
    Convert YYYYDDD integer/string values to pandas datetime.

    Example:
        2024032 -> 2024-02-01
    """
    s = series.astype(str).str.strip()
    year = s.str[:4]
    day_of_year = s.str[4:].astype(int) - 1
    return pd.to_datetime(year, format="%Y") + pd.to_timedelta(day_of_year, unit="D")


def rawtime2time(series: pd.Series) -> pd.Series:
    """
    Convert hour-decimal values to HH:MM:SS strings.

    Example:
        13.5 -> '13:30:00'
    """
    day_fractions = series.astype(float) / 24.0
    td = pd.to_timedelta(day_fractions, unit="D")

    return td.dt.components.apply(
        lambda x: f"{x['hours']:02}:{x['minutes']:02}:{x['seconds']:02}",
        axis=1,
    )