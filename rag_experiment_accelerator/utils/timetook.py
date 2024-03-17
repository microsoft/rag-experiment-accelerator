import time
from .logging import get_logger


class TimeTook(object):
    """
    Calculates the time a block took to run.
    Example usage:
    with TimeTook("sample"):
        s = [x for x in range(10000000)]
    Modified from: https://blog.usejournal.com/how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8 # noqa
    """

    def __init__(self, description, logger):
        self.description = description
        self.logger = logger if logger else get_logger(__name__)
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.info(f"Starting {self.description}")

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        self.logger.info(
            f"Time took for {self.description}: " f"{self.end - self.start} seconds"
        )
