from abc import ABC, abstractmethod


class Writer(ABC):
    @abstractmethod
    def write(self, path: str, data, **kwargs):
        pass

    @abstractmethod
    def copy(self, src: str, dest: str, **kwargs):
        pass

    @abstractmethod
    def delete(self, src: str):
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass
