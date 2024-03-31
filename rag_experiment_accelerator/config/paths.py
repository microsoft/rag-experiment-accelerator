import os
import glob
from datetime import datetime

from rag_experiment_accelerator.config.config import Config


def get_all_file_paths(directory: str) -> list[str]:
    """
    Returns a list of all file paths in a directory listed recursively.
    """
    pattern = os.path.join(directory, "**", "*")
    return [file for file in glob.glob(pattern, recursive=True) if os.path.isfile(file)]


def formatted_datetime_suffix():
    """Return a suffix to use when naming the run and its artifacts."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def mlflow_run_name(config: Config, suffix: str = None):
    """Returns a name to use for the MlFlow experiment run."""
    if not suffix:
        suffix = formatted_datetime_suffix()
    return f"{config.JOB_NAME}_{suffix}"
