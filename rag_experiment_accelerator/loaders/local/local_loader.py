from abc import abstractmethod
import os

from rag_experiment_accelerator.loaders.loader import Loader


class LocalLoader(Loader):
    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        pass

    def exists(self, src: str) -> bool:
        if os.path.exists(src):
            return True
        return False
