from requests import HTTPError
from sentence_transformers import SentenceTransformer
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    _size_model_mapping = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "bert-large-nli-mean-tokens": 1024
    }

    def __init__(self, model_name: str, dimension: int = None) -> None:
        if dimension is None:
            dimension = self._size_model_mapping.get(model_name)
            if dimension is None:
                raise ValueError(f"Dimension not provided and model name {model_name} not found in mapping")
        super().__init__(model_name=model_name, dimension=dimension)
        self._model = self.try_retrieve_model(model_name)


    def generate_embedding(self, chunk: str) -> list:
        return self._model.encode([str(chunk)]).tolist()

    
    def try_retrieve_model(self, tags: list[str] = None):
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"Retrieved model {self.model_name}")
            return model
        except HTTPError as e:
            raise e
        