import json

from rag_experiment_accelerator.io.local.writers.local_writer import LocalWriter
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class JsonlWriter(LocalWriter):
    """
    A class for writing data to a JSONL file.

    Inherits from the LocalWriter class.

    Attributes:
        None

    Methods:
        write_file: Writes data to a JSONL file.

    """

    def _write_file(self, path: str, data, **kwargs):
        """
        Writes the given data to a JSONL file.

        Args:
            path (str): The path to the JSONL file.
            data: The data to be written to the file.
            **kwargs: Additional keyword arguments to be passed to the json.dumps() function.

        Returns:
            None

        """
        with open(path, "a") as file:
            file.write(json.dumps(data, **kwargs) + "\n")
