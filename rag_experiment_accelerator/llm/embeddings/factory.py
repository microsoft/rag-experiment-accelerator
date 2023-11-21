from typing import Optional
from rag_experiment_accelerator.llm.embeddings.openai_embedding import OpenAIEmbeddingModel
from rag_experiment_accelerator.llm.embeddings.sentence_transformer_embedding import SentenceTransformerEmbeddingModel
from rag_experiment_accelerator.config.auth import OpenAICredentials


class EmbeddingModelFactory:
    """
    Factory class for creating embedding models based on the specified embedding type.
    """

    @staticmethod
    def create(embedding_type: str, model_name: str, dimension: int, openai_creds: Optional[OpenAICredentials]):
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
        if embedding_type == "openai":
            return OpenAIEmbeddingModel(model_name, openai_creds, dimension)
        elif embedding_type == "huggingface":
            return SentenceTransformerEmbeddingModel(model_name, dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of 'openai', 'huggingface'")
        
