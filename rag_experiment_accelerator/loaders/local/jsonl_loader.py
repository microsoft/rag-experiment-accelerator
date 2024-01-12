import json
import os
import pathlib
from rag_experiment_accelerator.loaders.local.local_loader import LocalLoader


class JsonlLoader(LocalLoader):
    """
    A class for loading data from a JSONL file.

    Inherits from the LocalLoader class.

    Attributes:
        None

    Methods:
        load(path: str, **kwargs) -> list: Loads data from a JSONL file and returns a list of loaded data.
        can_handle(path: str): Checks if the loader can handle the given file path.

    """

    def load(self, path: str, **kwargs) -> list:
        """
        Loads data from a JSONL file and returns a list of loaded data.

        Args:
            path (str): The path to the JSONL file.
            **kwargs: Additional keyword arguments to be passed to the json.loads() function.

        Returns:
            list: A list of loaded data.

        """
        data_load = []
        if os.path.exists(path):
            with open(path, "r") as file:
                for line in file:
                    data = json.loads(line, **kwargs)
                    data_load.append(data)
        return data_load

    def can_handle(self, path: str):
        """
        Checks if the loader can handle the given file path.

        Args:
            path (str): The file path to be checked.

        Returns:
            bool: True if the loader can handle the file path, False otherwise.

        """
        ext = pathlib.Path(path).suffix
        return ext == ".jsonl"
