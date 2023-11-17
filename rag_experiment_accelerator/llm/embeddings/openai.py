import openai
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger
logger = get_logger(__name__)

class OpenAIEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str, dimension: int) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(model_name=model_name, dimension=dimension)

    def generate_embedding(self, chunk: str) -> list[float]:
        params = {
            "input": [chunk],
            "model": self.model_name
        }

        embedding = openai.Embedding.create(**params)["data"][0]["embedding"]
        return [embedding]
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def try_retrieve_model(self):
        """
        Tries to retrieve a specified model from OpenAI.

        Args:
            model_name (str): The name of the model to retrieve.
            tags (list[str]): A list of capability tags to check for.
        Returns:
            openai.Model: The retrieved model object if successful.

        Raises:
            ValueError: If the model is not ready or does not have the required capabilities.
            openai.error.InvalidRequestError: If the model does not exist.
        """
        try:
            logger.info(f"Trying to retrieve model {self.model_name}")
            return openai.Model.retrieve(self.model_name)
        except openai.error.InvalidRequestError:
            logger.critical(f"Model {self.model_name} does not exist.")
            raise ValueError(f"Model {self.model_name} does not exist.")

# o = OpenAIEmbeddingModel("text-embedding-ada-002")
# o.try_retrieve_model()