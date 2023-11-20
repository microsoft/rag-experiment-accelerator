import openai
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel
from rag_experiment_accelerator.llm.openai_model import OpenAIModel
from rag_experiment_accelerator.config.auth import OpenAICredentials


class OpenAIEmbeddingModel(EmbeddingModel, OpenAIModel):

    def __init__(self, model_name: str, openai_creds: OpenAICredentials, dimension: int = None) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(model_name=model_name, dimension=dimension, tags=["embeddings", "inference"], openai_creds=openai_creds)
        
    def generate_embedding(self, chunk: str) -> list[float]:
        params = {
            "input": [chunk],
            "engine": self.model_name
        }

        self._openai_creds.set_credentials()
        embedding = openai.Embedding.create(**params)["data"][0]["embedding"]
        return [embedding]
