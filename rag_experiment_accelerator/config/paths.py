import os
import glob
from datetime import datetime

from rag_experiment_accelerator.config import Config  # noqa: E402


def get_all_files(directory):
    pattern = os.path.join(directory, "**", "*")
    all_files = [
        file for file in glob.glob(pattern, recursive=True) if os.path.isfile(file)
    ]
    return all_files


def formatted_datetime_suffix():
    """Return a suffix to use when naming the run and its artifacts."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def mlflow_run_name(config: Config, suffix: str = None):
    """Returns a name to use for the MlFlow experiment run."""
    if not suffix:
        suffix = formatted_datetime_suffix()
    return f"{config.NAME_PREFIX}_{suffix}"
