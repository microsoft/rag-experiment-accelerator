from abc import abstractmethod
import pathlib

from rag_experiment_accelerator.io.loader import Loader
from rag_experiment_accelerator.io.local.local_io_base import LocalIOBase


class LocalLoader(LocalIOBase, Loader):
    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        pass

    def _get_file_ext(self, path: str):
        return pathlib.Path(path).suffix
