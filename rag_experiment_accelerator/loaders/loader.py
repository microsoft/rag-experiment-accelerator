from abc import ABC, abstractmethod


class Loader(ABC):
    """
    Abstract base class for loaders.
    """

    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        """
        Load data from the given source.

        Args:
            src (str): The source of the data.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The loaded data.
        """
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        """
        Check if the loader can handle the given source.

        Args:
            src (str): The source to check.

        Returns:
            bool: True if the loader can handle the source, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, src: str) -> bool:
        """
        Check if the given source exists.

        Args:
            src (str): The source to check.

        Returns:
            bool: True if the source exists, False otherwise.
        """
        pass
