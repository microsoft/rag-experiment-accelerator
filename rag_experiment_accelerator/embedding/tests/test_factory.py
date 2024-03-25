from unittest.mock import patch, MagicMock
import pytest

from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel
from rag_experiment_accelerator.embedding.factory import create_embedding_model


def test_create_aoai_embedding_model():
    embedding_type = "azure"
    model_name = "test_model"
    dimension = 768
    environment = MagicMock()
    model = create_embedding_model(
        model_type=embedding_type,
        deployment_name=model_name,
        dimension=dimension,
        environment=environment,
    )
    assert isinstance(model, AOAIEmbeddingModel)


@patch("rag_experiment_accelerator.embedding.st_embedding_model.SentenceTransformer")
def test_create_st_embedding_model(mock_sentence_transformer):
    embedding_type = "sentence-transformer"
    model_name = "all-mpnet-base-v2"
    dimension = 768
    environment = MagicMock()
    model = create_embedding_model(
        model_type=embedding_type,
        model_name=model_name,
        dimension=dimension,
        environment=environment,
    )
    assert isinstance(model, STEmbeddingModel)


def test_create_raises_invalid_embedding_type():
    embedding_type = "not-valid"
    model_name = "test_model"
    dimension = 768
    environment = MagicMock()
    with pytest.raises(ValueError):
        create_embedding_model(
            model_type=embedding_type,
            model_name=model_name,
            dimension=dimension,
            environment=environment,
        )
