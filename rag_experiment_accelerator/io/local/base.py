import os


class LocalIOBase:
    """
    Base class for local input/output operations.
    """

    def exists(self, path: str) -> bool:
        """
        Check if a file or directory exists at the given path.

        Args:
            path (str): The path to check.

        Returns:
            bool: True if the file or directory exists, False otherwise.
        """
        if os.path.exists(path):
            return True
        return False
