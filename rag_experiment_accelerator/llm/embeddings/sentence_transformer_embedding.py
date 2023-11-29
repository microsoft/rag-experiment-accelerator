from requests import HTTPError
from sentence_transformers import SentenceTransformer
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """
    A class representing a Sentence Transformer Embedding Model.

    Args:
        model_name (str): The name of the Sentence Transformer model.
        dimension (int, optional): The dimension of the embeddings. If not provided, it will be determined based on the model name.

    Attributes:
        _size_model_mapping (dict): A mapping of model names to their corresponding dimensions.

    Methods:
        __init__(self, model_name: str, dimension: int = None) -> None: Initializes the SentenceTransformerEmbeddingModel object.
        generate_embedding(self, chunk: str) -> list: Generates the embedding for a given chunk of text.
        try_retrieve_model(self, tags: list[str] = None): Tries to retrieve the Sentence Transformer model.

    """

    _size_model_mapping = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "bert-large-nli-mean-tokens": 1024,
    }

    def __init__(self, model_name: str, dimension: int = None) -> None:
        """
        Initializes the SentenceTransformerEmbeddingModel object.

        Args:
            model_name (str): The name of the Sentence Transformer model.
            dimension (int, optional): The dimension of the embeddings. If not provided, it will be determined based on the model name.

        Raises:
            ValueError: If dimension is not provided and model name is not found in the mapping.

        """
        if dimension is None:
            dimension = self._size_model_mapping.get(model_name)
            if dimension is None:
                raise ValueError(
                    f"Dimension not provided and model name {model_name} not found in mapping"
                )
        super().__init__(model_name=model_name, dimension=dimension)
        self._model = self.try_retrieve_model()

    def generate_embedding(self, chunk: str) -> list:
        """
        Generates the embedding for a given chunk of text.

        Args:
            chunk (str): The input text.

        Returns:
            list: The embedding of the input text.

        """
        return self._model.encode([str(chunk)]).tolist()

    def try_retrieve_model(self):
        """
        Tries to retrieve the Sentence Transformer model.

        Args:
            tags (list[str], optional): Tags to filter the models. Defaults to None.

        Returns:
            The retrieved Sentence Transformer model.

        Raises:
            HTTPError: If there is an error retrieving the model.

        """
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"Retrieved model {self.model_name}")
            return model
        except Exception as e:
            logger.debug(e)
            raise ValueError(f"Unable to retrieve Sentence Transformer model with name: {self.model_name}")
