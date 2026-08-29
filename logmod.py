"""Logging configuration and setup module."""

import datetime as dt
import logging
import os
import sys
from typing import Optional, Union


def logs(
    show_level: Optional[Union[int, str]] = "info",
    save_level: Optional[Union[int, str]] = None,
    program_name: Optional[str] = None,
    path: Optional[str] = None,
    threads: bool = False,
    multiproc: bool = False,
    show_color: bool = True,
) -> None:
    """Configures global console and file logging handlers.

    Args:
        show_level: Minimum log level for standard output ('debug', 'info', 'warning', 'error', etc.).
        save_level: Minimum log level to persist to disk.
        program_name: Optional application name included in the output log filename.
        path: Directory path where log files will be stored. Defaults to '_logs'.
        threads: Whether to include thread names in log formatting.
        multiproc: Whether to include process names in log formatting.
        show_color: Whether to attempt colored console output if coloredlogs is installed.
    """
    logger_root = logging.getLogger()
    fmt_items = (
        "%(asctime)s",
        "%(levelname)-8s",
        "%(threadName)s" if threads else None,
        "%(processName)s" if multiproc else None,
        "%(name)s",
        "%(message)s",
    )
    fmt = " - ".join((item for item in fmt_items if item is not None))
    formatter = logging.Formatter(fmt)
    logging.addLevelName(5, "VERBOSE")
    logger_root.setLevel(5)

    if show_level and show_color:
        try:
            import coloredlogs  # type: ignore

            if isinstance(show_level, str):
                show_level = show_level.upper()
            coloredlogs.install(level=show_level, fmt=fmt)
        except ImportError:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            if isinstance(show_level, str):
                show_level = getattr(logging, show_level.upper(), logging.INFO)
            stream_handler.setLevel(show_level)
            logger_root.addHandler(stream_handler)
    elif show_level:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        if isinstance(show_level, str):
            show_level = getattr(logging, show_level.upper(), logging.INFO)
        stream_handler.setLevel(show_level)
        logger_root.addHandler(stream_handler)

    if save_level:
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "_logs")
        os.makedirs(path, exist_ok=True)
        now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = f"{program_name}_" if program_name else ""
        log_file = os.path.join(path, f"{prefix}{now}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        if isinstance(save_level, str):
            save_level = getattr(logging, save_level.upper(), logging.INFO)
        file_handler.setLevel(save_level)
        logger_root.addHandler(file_handler)
