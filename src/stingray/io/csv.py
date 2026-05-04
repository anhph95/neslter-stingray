# io/csv.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import concurrent.futures
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_parallel(
    file_list: Iterable[str | Path],
    max_workers: int | None = None,
    strict: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read multiple CSV files in parallel and concatenate into a single DataFrame.

    Parameters
    ----------
    file_list : iterable of str or Path
        Files to read.
    max_workers : int, optional
        Number of threads. Defaults to min(32, cpu_count()).
    strict : bool
        If True, raise on first failure instead of skipping files.
    **read_csv_kwargs
        Passed directly to pandas.read_csv.

    Returns
    -------
    DataFrame
    """

    files = [Path(f) for f in file_list]

    if len(files) == 0:
        return pd.DataFrame()

    def safe_read_csv(file: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(file, low_memory=False, **read_csv_kwargs)
        except Exception as e:
            if strict:
                raise
            logger.warning(f"Failed to read {file}: {e}")
            return pd.DataFrame()

    # Better default: bounded threads
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(safe_read_csv, files))

    # Filter empty frames
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        logger.warning("No CSV files were successfully read.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)