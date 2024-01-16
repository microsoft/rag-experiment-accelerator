import pytest

from rag_experiment_accelerator.config.credentials import OpenAICredentials
from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel
from rag_experiment_accelerator.embedding.factory import EmbeddingModelFactory
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel


def test_create_aoai_embedding_model():
    embedding_type = "azure"
    model_name = "test_model"
    dimension = 768
    openai_creds = OpenAICredentials("open_ai", "", "", "")
    model = EmbeddingModelFactory.create(
        type=embedding_type,
        deployment_name=model_name,
        dimension=dimension,
        openai_creds=openai_creds,
    )
    assert isinstance(model, AOAIEmbeddingModel)


def test_create_st_embedding_model():
    embedding_type = "sentence-transformer"
    model_name = "all-mpnet-base-v2"
    dimension = 768
    model = EmbeddingModelFactory.create(
        type=embedding_type, model_name=model_name, dimension=dimension
    )
    assert isinstance(model, STEmbeddingModel)


def test_create_raises_invalid_embedding_type():
    embedding_type = "not-valid"
    model_name = "test_model"
    dimension = 768
    with pytest.raises(ValueError):
        EmbeddingModelFactory.create(
            type=embedding_type,
            model_name=model_name,
            dimension=dimension,
        )
