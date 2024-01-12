from abc import ABC, abstractmethod


class Writer(ABC):
    """Abstract base class for writers."""

    @abstractmethod
    def write(self, path: str, data):
        """Write data to the specified path.

        Args:
            path (str): The path to write the data to.
            data: The data to be written.
        """
        pass

    @abstractmethod
    def copy(self, src: str, dest: str, **kwargs):
        """Copy a file from the source path to the destination path.

        Args:
            src (str): The source path of the file to be copied.
            dest (str): The destination path where the file should be copied to.
            **kwargs: Additional keyword arguments for the copy operation.
        """
        pass

    @abstractmethod
    def delete(self, src: str):
        """Delete the file at the specified path.

        Args:
            src (str): The path of the file to be deleted.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists at the specified path.

        Args:
            path (str): The path to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        pass
