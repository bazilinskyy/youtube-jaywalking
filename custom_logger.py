"""Custom logger module supporting brace-style formatting."""

import logging
from typing import Any


class CustomLogger:
    """Logger wrapper that handles brace-style string formatting.

    Wraps a standard logging.Logger instance to allow str.format() style
    positional placeholders within log messages while retaining standard logging levels.

    Args:
        name: Name of the logger, typically __name__.
    """

    def __init__(self, name: str) -> None:
        """Initializes CustomLogger with the specified name."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(5)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level DEBUG."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level INFO."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level WARNING."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level ERROR."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level CRITICAL."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with the specified integer level."""
        if self.logger.isEnabledFor(level):
            if args:
                try:
                    msg = msg.format(*args)
                except Exception:
                    pass
            self.logger._log(level, msg, args=(), **kwargs)
