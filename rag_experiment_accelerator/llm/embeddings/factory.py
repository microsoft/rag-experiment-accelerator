from rag_experiment_accelerator.llm.embeddings.openai_embedding import OpenAIEmbeddingModel
from rag_experiment_accelerator.llm.embeddings.sentence_transformer_emebedding import SentenceTransformerEmbeddingModel
from rag_experiment_accelerator.config.auth import OpenAICredentials


class EmbeddingModelFactory:
    @staticmethod
    def create(embedding_type: str, model_name: str, dimension: int, openai_creds: OpenAICredentials):
        if embedding_type == "openai":
            return OpenAIEmbeddingModel(model_name, openai_creds, dimension)
        elif embedding_type == "huggingface":
            return SentenceTransformerEmbeddingModel(model_name, dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of 'openai', 'huggingface'")
        
