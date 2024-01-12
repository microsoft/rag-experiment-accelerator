from abc import abstractmethod
import os
import pathlib
import shutil

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.writers.writer import Writer


logger = get_logger(__name__)


class LocalWriter(Writer):
    def _try_make_dir(self, dir: str, exist_ok: bool = True):
        try:
            os.makedirs(dir, exist_ok=exist_ok)
        except Exception as e:
            logger.error(
                f"Unable to create the directory: {dir}. Please ensure"
                " you have the proper permissions to create the directory."
            )
            raise e

    def _prepare_write(self, dir: str):
        self._try_make_dir(dir)

    @abstractmethod
    def write_file():
        pass

    def write(self, path: str, data, **kwargs):
        dir = pathlib.Path(path).parent
        self._prepare_write(dir)
        self.write_file(path, data, **kwargs)

    def copy(self, src: str, dest: str, **kwargs):
        dest_dir = pathlib.Path(dest).parent
        self._prepare_write(dest_dir)
        shutil.copyfile(src, dest, **kwargs)

    def delete(self, src: str):
        if os.path.exists(src):
            os.remove(src)

    def list_filenames(self, dir: str):
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    def exists(self, path: str) -> bool:
        if os.path.exists(path):
            return True
        return False
