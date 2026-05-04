from __future__ import annotations

import numpy as np

def convert_gps(raw_series, sign_series):
    signs = sign_series.map({'N': 1, 'S': -1, 'E': 1, 'W': -1})
    return signs * (np.floor(raw_series / 100) + (raw_series % 100) / 60)