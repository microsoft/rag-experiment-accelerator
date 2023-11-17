import openai
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

class AzureOpenAIEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str, dimension: int = 1536) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(model_name=model_name, dimension=dimension)

    def generate_embedding(self, chunk: str) -> list[float]:
        params = {
            "input": [chunk],
            "engine": self.model_name
        }

        embedding = openai.Embedding.create(**params)["data"][0]["embedding"]
        return [embedding]
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def try_retrieve_model(self):
        tags=["embeddings", "inference"]
        try:
            model = openai.Model.retrieve(self.model_name)
            if model["status"] != "succeeded":
                logger.critical(f"Model {self.model_name} is not ready.")
                raise ValueError(f"Model {self.model_name} is not ready.")
            for tag in tags:
                if not model["capabilities"][tag]:
                    logger.critical(
                        f"Model {self.model_name} does not have the {tag} capability."
                    )
                    raise ValueError(
                        f"Model {self.model_name} does not have the {tag} capability."
                    )
            return model
        except openai.error.InvalidRequestError:
            logger.critical(f"Model {self.model_name} does not exist.")
            raise ValueError(f"Model {self.model_name} does not exist.")