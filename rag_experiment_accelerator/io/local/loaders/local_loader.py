from abc import abstractmethod
import pathlib

from rag_experiment_accelerator.io.loader import Loader
from rag_experiment_accelerator.io.local.base import LocalIOBase


class LocalLoader(LocalIOBase, Loader):
    """
    A class that represents a local data loader.

    This class provides methods for loading data from a local source.

    Attributes:
        None

    Methods:
        load(src: str, **kwargs) -> list:
            Abstract method to load data from a local source.

        can_handle(src: str) -> bool:
            Abstract method to check if the loader can handle the given source.

        _get_file_ext(path: str):
            Internal method to get the file extension from a given path.
    """

    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        """
        Abstract method to load data from a local source.

        Args:
            src (str): The path or source of the data.

        Returns:
            list: The loaded data.
        """
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        """
        Abstract method to check if the loader can handle the given source.

        Args:
            src (str): The path or source of the data.

        Returns:
            bool: True if the loader can handle the source, False otherwise.
        """
        pass

    def _get_file_ext(self, path: str):
        """
        Internal method to get the file extension from a given path.

        Args:
            path (str): The path of the file.

        Returns:
            str: The file extension.
        """
        return pathlib.Path(path).suffix
