from rag_experiment_accelerator.llm.embeddings.openai_embeddings import OpenAIEmbeddingsModel
from rag_experiment_accelerator.llm.embeddings.sentence_transformer_emebeddings import SentenceTransformerEmbeddingsModel
from rag_experiment_accelerator.config.auth import OpenAICredentials


class EmbeddingsModelFactory:
    @staticmethod
    def create(embedding_type: str, model_name: str, dimension: int, openai_creds: OpenAICredentials):
        if embedding_type == "openai":
            return OpenAIEmbeddingsModel(model_name, openai_creds, dimension)
        elif embedding_type == "huggingface":
            return SentenceTransformerEmbeddingsModel(model_name, dimension)
        else:
            raise ValueError(f"Invalid embedding type: {type}. Must be one of 'openai', 'huggingface'")
        
