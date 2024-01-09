from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel


class EmbeddingModelFactory:
    """
    Factory class for creating embedding models based on the specified embedding type.
    """

    @staticmethod
    def create(type: str, **kwargs):
        """
        Create an embedding model based on the specified type.

        Args:
            type (str): The type of embedding model to create. Must be one of ['azure', 'sentence-transformer'].
            **kwargs: Additional keyword arguments to be passed to the embedding model constructor.

        Returns:
            An instance of the specified embedding model.

        Raises:
            ValueError: If an invalid embedding type is provided.
        """
        if type == "azure":
            return AOAIEmbeddingModel(**kwargs)
        elif type == "sentence-transformer":
            return STEmbeddingModel(**kwargs)
        else:
            raise ValueError(
                f"Invalid embedding type: {type}. Must be one of ['azure', 'sentence-transformer']"
            )
