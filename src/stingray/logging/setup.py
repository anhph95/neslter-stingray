# src/stingray/logging/setup.py

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: str | Path = "logs",
    name: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Configure global logging for the Stingray package.

    This attaches handlers to the *root logger* so all modules using:
        logging.getLogger(__name__)
    automatically use the same configuration.

    Returns
    -------
    logging.Logger
        Root logger
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: use ROOT logger, not named logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent duplicate handlers (important for notebooks / repeated runs)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger_name = name or "stingray"

    if file:
        out_path = log_dir / f"{logger_name}_{ts}.out.log"
        err_path = log_dir / f"{logger_name}_{ts}.err.log"

        out_handler = logging.FileHandler(out_path)
        out_handler.setLevel(logging.INFO)
        out_handler.setFormatter(fmt)

        err_handler = logging.FileHandler(err_path)
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(fmt)

        logger.addHandler(out_handler)
        logger.addHandler(err_handler)

        logger.info("Logging to: %s", out_path)
        logger.info("Errors to: %s", err_path)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    return logger