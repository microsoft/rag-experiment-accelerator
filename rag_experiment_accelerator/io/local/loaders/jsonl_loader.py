import json

from rag_experiment_accelerator.io.local.loaders.local_loader import LocalLoader


class JsonlLoader(LocalLoader):
    """A class for loading data from JSONL files."""

    def load(self, path: str, **kwargs) -> list:
        """Load data from a JSONL file.

        Args:
            path (str): The path to the JSONL file.
            **kwargs: Additional keyword arguments to be passed to json.loads().

        Returns:
            list: A list of loaded data.

        Raises:
            FileNotFoundError: If the file is not found at the specified path.
        """
        if not self.exists(path):
            raise FileNotFoundError(f"File not found at path: {path}")

        data_load = []
        with open(path, "r") as file:
            for line in file:
                data = json.loads(line, **kwargs)
                data_load.append(data)

        return data_load

    def can_handle(self, path: str) -> bool:
        """Check if the loader can handle the given file path.

        Args:
            path (str): The file path to check.

        Returns:
            bool: True if the loader can handle the file, False otherwise.
        """
        ext = self._get_file_ext(path)
        return ext == ".jsonl"
