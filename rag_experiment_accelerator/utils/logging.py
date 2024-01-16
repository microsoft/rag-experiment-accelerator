import logging
import os

# Global variable to cache the logging level
_cached_logging_level = None


def set_logging_params():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def get_logger(name: str) -> logging.Logger:
    set_logging_params()

    global _cached_logging_level
    if _cached_logging_level is None:
        _cached_logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(_cached_logging_level)

    return logger
