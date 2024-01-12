from abc import abstractmethod
import os
import pathlib
import shutil

from rag_experiment_accelerator.io.local.local_io_base import LocalIOBase
from rag_experiment_accelerator.io.writer import Writer
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class LocalWriter(LocalIOBase, Writer):
    def _make_dir(self, dir: str):
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Unable to create the directory: {dir}. Please ensure"
                " you have the proper permissions to create the directory."
            )
            raise e

    def _get_dirname(self, path: str):
        return pathlib.Path(path).parent

    @abstractmethod
    def write_file(path: str, data, **kwargs):
        pass

    def write(self, path: str, data, **kwargs):
        dir = self._get_dirname(path)
        self._make_dir(dir)
        try:
            self.write_file(path, data, **kwargs)
        except Exception as e:
            logger.error(
                f"Unable to write to file to path: {path}. Please ensure"
                " you have the proper permissions to write to the file."
            )
            raise e

    def copy(self, src: str, dest: str, **kwargs):
        if not self.exists(src):
            raise FileNotFoundError(f"Source file {src} does not exist.")

        dest_dir = self._get_dirname(dest)
        # make dest dir if it doesn't exist
        self._make_dir(dest_dir)
        try:
            shutil.copyfile(src, dest, **kwargs)
        except Exception as e:
            logger.error(
                f"Unable to copy file from {src} to {dest}. Please ensure"
                " you have the proper permissions to copy the file."
            )
            raise e

    def delete(self, src: str):
        if self.exists(src):
            os.remove(src)

    def list_filenames(self, dir: str):
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
