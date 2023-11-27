from abc import abstractmethod
from rag_experiment_accelerator.llm.base import LLMModel


class EmbeddingModel(LLMModel):
    """
    Base class for embedding models.
    Args:
        model_name (str): The name of the embedding model.
        dimension (int): The dimension of the embeddings.
    Attributes:
        model_name (str): The name of the embedding model.
        dimension (int): The dimension of the embeddings.
    Methods:
        generate_embedding(chunk: str) -> list: Abstract method to generate embeddings for a given chunk of text.
    """

    def __init__(self, model_name: str, dimension: int, *args, **kwargs) -> None:
        super().__init__(model_name=model_name, *args, **kwargs)
        self.dimension = dimension

    @abstractmethod
    def generate_embedding(self, chunk: str) -> list:
        """
        abstract method to generate embeddings for a given chunk of text.
        Args:
            chunk (str): The input text chunk for which the embedding needs to be generated.
        Returns:
            list: The generated embedding as a list.
        """
        pass
