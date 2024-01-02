from typing import Optional
from rag_experiment_accelerator.credentials.openai_credentials import OpenAICredentials
from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel


class EmbeddingModelFactory:
    """
    Factory class for creating embedding models based on the specified embedding type.
    """

    @staticmethod
    def create(embedding_type: str, model_name: Optional[str], dimension: Optional[int], openai_creds: Optional[OpenAICredentials]):
        """
        Create an embedding model based on the specified embedding type.

        Args:
            embedding_type (str): The type of embedding model to create. Must be one of 'openai' or 'huggingface'.
            model_name (str): The name of the model.
            dimension (int): The dimension of the embedding.
            openai_creds (OpenAICredentials): The OpenAI credentials.

        Returns:
            An instance of the embedding model based on the specified embedding type.

        Raises:
            ValueError: If the specified embedding type is invalid.
        """
        if embedding_type == "azure":
            return AOAIEmbeddingModel(deployment_name=model_name, creds=openai_creds, dimension=dimension)
        elif embedding_type == "sentence-transformer":
            return STEmbeddingModel(model_name=model_name, dimension=dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of ['azure', 'sentence-transformer']")