from abc import ABC, abstractmethod

class LLMModelBase(ABC):

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # self._tags = tags

    @abstractmethod
    def try_retrieve_model(self):
       pass


class EmbeddingModel(LLMModelBase):

    def __init__(self, model_name: str, dimension: int, *args, **kwargs) -> None:
        super().__init__(model_name=model_name, *args, **kwargs)
        self.dimension = dimension
        
    @abstractmethod
    def generate_embedding(self, chunk: str) -> list[float]:
        pass
