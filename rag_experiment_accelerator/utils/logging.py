import logging
import os
import sys

# Global variable to cache the logging level
_cached_logging_level = None


def get_logger(name: str) -> logging.Logger:
    """Get Logger

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: named logger
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    global _cached_logging_level
    if not _cached_logging_level:
        _cached_logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(_cached_logging_level)
    logger.addHandler(handler)

    return logger
