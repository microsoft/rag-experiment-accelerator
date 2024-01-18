from abc import ABC, abstractmethod


class Writer(ABC):
    """Abstract base class for a writer."""

    @abstractmethod
    def write(self, path: str, data, **kwargs):
        """Write data to a file.

        Args:
            path (str): The path of the file to write to.
            data: The data to write to the file.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def copy(self, src: str, dest: str, **kwargs):
        """Copy a file from source to destination.

        Args:
            src (str): The path of the source file.
            dest (str): The path of the destination file.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def delete(self, src: str):
        """Delete a file.

        Args:
            src (str): The path of the file to delete.

        Returns:
            None
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists.

        Args:
            path (str): The path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        pass
