from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel


def create_embedding_model(model_type: str, **kwargs):
    match model_type:
        case "azure":
            return AOAIEmbeddingModel(**kwargs)
        case "sentence-transformer":
            return STEmbeddingModel(**kwargs)
        case _:
            raise ValueError(
                f"Invalid embedding type: {model_type}. Must be one of ['azure', 'sentence-transformer']"
            )
