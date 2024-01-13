from abc import abstractmethod
import os
import pathlib
import shutil
from rag_experiment_accelerator.io.exceptions import CopyException, WriteException

from rag_experiment_accelerator.io.local.base import LocalIOBase
from rag_experiment_accelerator.io.writer import Writer
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class LocalWriter(LocalIOBase, Writer):
    """
    A class that provides methods for writing files locally.

    Inherits from LocalIOBase and Writer.
    """

    def _make_dir(self, dir: str):
        """
        Creates a directory if it doesn't exist.

        Args:
            dir (str): The directory path.
        """
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Unable to create the directory: {dir}. Please ensure"
                " you have the proper permissions to create the directory."
            )
            raise e

    def _get_dirname(self, path: str):
        """
        Returns the parent directory of a given path.

        Args:
            path (str): The file path.

        Returns:
            str: The parent directory path.
        """
        return pathlib.Path(path).parent

    @abstractmethod
    def _write_file(path: str, data, **kwargs):
        """
        Abstract method for writing a file.

        Args:
            path (str): The file path.
            data: The data to be written to the file.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        pass

    def write(self, path: str, data, **kwargs):
        """
        Writes data to a file at the specified path.

        Args:
            path (str): The file path.
            data: The data to be written to the file.
            **kwargs: Additional keyword arguments.

        Raises:
            Exception: If unable to write to the file.
        """
        dir = self._get_dirname(path)
        self._make_dir(dir)
        try:
            self._write_file(path, data, **kwargs)
        except Exception as e:
            raise WriteException(path, e)

    def copy(self, src: str, dest: str, **kwargs):
        """
        Copies a file from the source path to the destination path.

        Args:
            src (str): The source file path.
            dest (str): The destination file path.
            **kwargs: Additional keyword arguments.

        Raises:
            FileNotFoundError: If the source file does not exist.
            Exception: If unable to copy the file.
        """
        if not self.exists(src):
            raise FileNotFoundError(f"Source file {src} does not exist.")

        dest_dir = self._get_dirname(dest)
        # make dest dir if it doesn't exist
        self._make_dir(dest_dir)
        try:
            shutil.copyfile(src, dest, **kwargs)
        except Exception as e:
            raise CopyException(src, dest, e)

    def delete(self, src: str):
        """
        Deletes a file at the specified path.

        Args:
            src (str): The file path.

        Raises:
            Exception: If unable to delete the file.
        """
        if self.exists(src):
            os.remove(src)

    def list_filenames(self, dir: str):
        """
        Returns a list of filenames in the specified directory.

        Args:
            dir (str): The directory path.

        Returns:
            list: A list of filenames in the directory.
        """
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
