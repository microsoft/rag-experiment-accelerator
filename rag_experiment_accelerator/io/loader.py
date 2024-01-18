from abc import ABC, abstractmethod


class Loader(ABC):
    """
    Abstract base class for data loaders.
    """

    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        """
        Load data from the specified source.

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
        Check if the loader can handle the specified source.

        Args:
            src (str): The source to check.

        Returns:
            bool: True if the loader can handle the source, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, src: str) -> bool:
        """
        Check if the specified source exists.

        Args:
            src (str): The source to check.

        Returns:
            bool: True if the source exists, False otherwise.
        """
        pass
