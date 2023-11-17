from rag_experiment_accelerator.llm.embeddings.openai import OpenAIEmbeddingModel 
from rag_experiment_accelerator.llm.embeddings.sentence_transformers import SentenceTransformersEmbeddingModel
from rag_experiment_accelerator.llm.embeddings.azure_openai import AzureOpenAIEmbeddingModel


class EmbeddingModelFactory:
    @staticmethod
    def create(embedding_type: str, model_name: str, dimension: int, openai_api_type: str) -> None:
        if embedding_type == "openai":
            if openai_api_type == "azure":
                return AzureOpenAIEmbeddingModel(model_name, dimension)
            elif openai_api_type == "openai":
                return OpenAIEmbeddingModel(model_name, dimension)
        elif embedding_type == "huggingface":
            return SentenceTransformersEmbeddingModel(model_name, dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of 'openai', 'huggingface'")
        
