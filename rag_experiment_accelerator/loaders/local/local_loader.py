from abc import abstractmethod
import os

from rag_experiment_accelerator.loaders.loader import Loader


class LocalLoader(Loader):
    """
    A class representing a local data loader.

    This class provides methods to load data from a local source and check if the source exists.

    Attributes:
        None

    Methods:
        load(src: str, **kwargs) -> list: Abstract method to load data from a local source.
        can_handle(src: str) -> bool: Abstract method to check if the loader can handle the given source.
        exists(src: str) -> bool: Method to check if the source exists.

    """

    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        pass

    def exists(self, src: str) -> bool:
        """
        Check if the source exists.

        Args:
            src (str): The source path.

        Returns:
            bool: True if the source exists, False otherwise.

        """
        if os.path.exists(src):
            return True
        return False
