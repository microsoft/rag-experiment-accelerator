from abc import ABC, abstractmethod
import openai
from requests import HTTPError
from sentence_transformers import SentenceTransformer
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

class SentenceTransformersEmbeddingModel(EmbeddingModel):
    _size_model_mapping = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "bert-large-nli-mean-tokens": 1024
    }

    def __init__(self, model_name: str, dimension: int) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self._model = self.try_retrieve_model(model_name)

    def generate_embedding(self, chunk: str) -> list[float]:
        return self._model.encode([str(chunk)]).tolist()

    def get_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        return self._size_model_mapping[self.model_name]
    
    def try_retrieve_model(self, tags: list[str] = None):
        try:
            logger.info(f"Trying to retrieve model {self.model_name}")
            return SentenceTransformer(self.model_name)
        except HTTPError as e:
            raise e