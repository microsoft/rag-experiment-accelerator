from abc import abstractmethod
from rag_experiment_accelerator.llm.base import LLMModel


class EmbeddingModel(LLMModel):

    def __init__(self, model_name: str, dimension: int, *args, **kwargs) -> None:
        super().__init__(model_name=model_name, *args, **kwargs)
        self.dimension = dimension
        
    @abstractmethod
    def generate_embedding(self, chunk: str) -> list:
        pass