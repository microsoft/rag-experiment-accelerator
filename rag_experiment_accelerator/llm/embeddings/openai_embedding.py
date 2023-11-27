import openai
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel
from rag_experiment_accelerator.llm.openai_model import OpenAIModel
from rag_experiment_accelerator.config.config import OpenAICredentials


class OpenAIEmbeddingModel(EmbeddingModel, OpenAIModel):
    """
    A class representing an OpenAI embedding model.
    Args:
        model_name (str): The name of the model.
        openai_creds (OpenAICredentials): The OpenAI credentials.
        dimension (int, optional): The dimension of the embedding. Defaults to None.
    Attributes:
        model_name (str): The name of the model.
        dimension (int): The dimension of the embedding.
        tags (list[str]): The tags associated with the model.
        openai_creds (OpenAICredentials): The OpenAI credentials.
    Methods:
        generate_embedding(chunk: str) -> list[float]: Generates the embedding for a given chunk of text.
    """

    def __init__(
        self, model_name: str, openai_creds: OpenAICredentials, dimension: int = None
    ) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            tags=["embeddings", "inference"],
            openai_creds=openai_creds,
        )

    def generate_embedding(self, chunk: str) -> list[float]:
        """
        Generates the embedding for a given chunk of text.
        Args:
            chunk (str): The input text.
        Returns:
            list[float]: The generated embedding.
        """
        params = {
            "input": [chunk],
        }

        if self._openai_creds.OPENAI_API_TYPE == "azure":
            params["engine"] = self.model_name
        else:
            params["model"] = self.model_name

        embedding = openai.Embedding.create(**params)["data"][0]["embedding"]
        return [embedding]
