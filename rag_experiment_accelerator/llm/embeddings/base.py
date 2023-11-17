from abc import ABC, abstractmethod


class EmbeddingModel(ABC):

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension
        
    @abstractmethod
    def generate_embedding(self, chunk: str) -> list[float]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def try_retrieve_model(self):
       raise NotImplementedError
