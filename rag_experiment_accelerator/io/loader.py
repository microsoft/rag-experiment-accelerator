from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def load(self, src: str, **kwargs) -> list:
        pass

    @abstractmethod
    def can_handle(self, src: str) -> bool:
        pass

    @abstractmethod
    def exists(self, src: str) -> bool:
        pass
