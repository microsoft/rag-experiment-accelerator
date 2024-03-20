from sentence_transformers import SentenceTransformer
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class STEmbeddingModel(EmbeddingModel):
    """
    STEmbeddingModel is a class that represents a sentence transformer embedding model.

    Args:
        model_name (str): The name of the pre-trained model to use for embedding.
        dimension (int, optional): The dimension of the embedding. If not provided, it will be determined based on the model name.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.

    Attributes:
        _size_model_mapping (dict): A mapping of supported model names to their corresponding dimensions.

    Raises:
        ValueError: If the dimension is not provided and the model name is not found in the mapping.

    Methods:
        generate_embedding(chunk: str) -> list: Generates the embedding for a given chunk of text.

    """

    _size_model_mapping = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "bert-large-nli-mean-tokens": 1024,
    }

    def __init__(self, model_name: str, dimension: int = None, **kwargs) -> None:
        """
        Initializes an instance of the STEmbeddingModel class.

        Args:
            model_name (str): The name of the pre-trained model to use for embedding.
            dimension (int, optional): The dimension of the embedding. If not provided, it will be determined based on the model name.
            **kwargs: Additional keyword arguments to be passed to the base class constructor.

        Raises:
            ValueError: If the dimension is not provided and the model name is not found in the mapping.

        """
        if dimension is None:
            dimension = self._size_model_mapping.get(model_name)
            if dimension is None:
                raise ValueError(
                    f"Dimension not provided and model name {model_name} not found in mapping. Please provide a dimension or specify a supported model name in {self._size_model_mapping.keys()}"
                )
        super().__init__(name=model_name, dimension=dimension, **kwargs)
        try:
            self._model = SentenceTransformer(self.name)
        except OSError as e:
            logger.error(
                f"Error retrieving model: {self.name}. Please check that the model name is correct and that you have an internet connection."
            )
            raise e

    def generate_embedding(self, chunk: str) -> list[float]:
        """
        Generates the embedding for a given chunk of text.

        Args:
            chunk (str): The text to generate the embedding for.

        Returns:
            list: The generated embedding as a list.

        """
        return self._model.encode([str(chunk)]).tolist()[0]
