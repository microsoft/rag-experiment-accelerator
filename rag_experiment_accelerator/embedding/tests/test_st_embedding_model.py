from unittest.mock import patch
import pytest
import numpy as np
from rag_experiment_accelerator.embedding.st_embedding_model import STEmbeddingModel


@patch("rag_experiment_accelerator.embedding.st_embedding_model.SentenceTransformer")
def test_generate_embedding(mock_sentence_transformer):
    expected_embeddings = [0.1, 0.2, 0.3]
    mock_embeddings = np.array([expected_embeddings])
    mock_sentence_transformer.return_value.encode.return_value = mock_embeddings

    model = STEmbeddingModel("all-mpnet-base-v2")
    embeddings = model.generate_embedding("Hello world")

    assert expected_embeddings == embeddings


def test_sentence_transformer_embedding_model_raises_non_existing_model():
    with pytest.raises(OSError):
        STEmbeddingModel("non-existing-model", 123)


def test_sentence_transformer_embedding_model_raises_unsupported_model():
    with pytest.raises(ValueError):
        STEmbeddingModel("non-existing-model")


@patch("rag_experiment_accelerator.embedding.st_embedding_model.SentenceTransformer")
def test_sentence_transformer_embedding_model_succeeds(mock_sentence_transformer):
    try:
        STEmbeddingModel("all-mpnet-base-v2")
    except BaseException:
        assert False, "Should not have thrown an exception"
