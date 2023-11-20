import pytest
from rag_experiment_accelerator.llm.embeddings.factory import EmbeddingsModelFactory
from rag_experiment_accelerator.config.auth import OpenAICredentials


def test_create_openai_embedding_model():
    embedding_type = "openai"
    model_name = "test_model"
    dimension = 768
    openai_creds = OpenAICredentials("open_ai", "", "", "")
    model = EmbeddingsModelFactory.create(embedding_type=embedding_type, model_name=model_name, dimension=dimension, openai_creds=openai_creds)
    model.__class__ == 'OpenAIEmbeddingModel'


def test_create_huggingface_embedding_model():
    embedding_type = "huggingface"
    model_name = "all-mpnet-base-v2"
    dimension = 768
    openai_creds = OpenAICredentials("open_ai", "", "", "")
    model = EmbeddingsModelFactory.create(embedding_type=embedding_type, model_name=model_name, dimension=dimension, openai_creds=openai_creds)
    model.__class__ == 'SentenceTransformersEmbeddingModel'


def test_create_raise_invalid_embedding_type():
    embedding_type = "not-valid"
    model_name = "test_model"
    dimension = 768
    openai_creds = OpenAICredentials("open_ai", "", "", "")
    with pytest.raises(ValueError):
        EmbeddingsModelFactory.create(embedding_type=embedding_type, model_name=model_name, dimension=dimension, openai_creds=openai_creds)