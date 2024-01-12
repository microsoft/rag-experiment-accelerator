import json

from rag_experiment_accelerator.writers.local.local_writer import LocalWriter


class JsonlWriter(LocalWriter):
    """
    A class for writing data to a file in JSONL format.

    Inherits from the LocalWriter class.

    Methods:
        write_file(path: str, data, **kwargs): Writes the given data to a file in JSONL format.

    """

    def write_file(self, path: str, data, **kwargs):
        """
        Writes the given data to a file in JSONL format.

        Args:
            path (str): The path to the file.
            data: The data to be written to the file.
            **kwargs: Additional keyword arguments to be passed to the json.dumps function.

        """
        with open(path, "a") as file:
            file.write(json.dumps(data, **kwargs) + "\n")
