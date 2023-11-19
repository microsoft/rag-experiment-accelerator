from rag_experiment_accelerator.llm.openai import OpenAIEmbeddingModel
from rag_experiment_accelerator.llm.sentence_transformers import SentenceTransformersEmbeddingModel
from rag_experiment_accelerator.utils.auth import OpenAICredentials


class EmbeddingModelFactory:
    @staticmethod
    def create(embedding_type: str, model_name: str, dimension: int, openai_creds: OpenAICredentials):
        if embedding_type == "openai":
            return OpenAIEmbeddingModel(model_name, openai_creds, dimension)
        elif embedding_type == "huggingface":
            return SentenceTransformersEmbeddingModel(model_name, dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of 'openai', 'huggingface'")
        
