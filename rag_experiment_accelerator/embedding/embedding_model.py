from abc import abstractmethod, ABC


class EmbeddingModel(ABC):
    """
    Base class for embedding models.

    Args:
        name (str): The name of the embedding model.
        dimension (int): The dimension of the embeddings.

    Attributes:
        dimension (int): The dimension of the embeddings.

    Methods:
        generate_embedding(chunk: str) -> list: Abstract method to generate embeddings for a given chunk of text.
    """

    def __init__(self, name: str, dimension: int, **kwargs) -> None:
        self.dimension = dimension
        self.name = name

    @abstractmethod
    def generate_embedding(self, chunk: str) -> list[float]:
        """
        abstract method to generate embeddings for a given chunk of text.

        Args:
            chunk (str): The input text chunk for which the embedding needs to be generated.

        Returns:
            list: The generated embedding as a list.
        """
        pass
