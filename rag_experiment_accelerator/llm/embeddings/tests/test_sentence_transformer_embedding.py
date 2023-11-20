from unittest.mock import patch
import pytest
import numpy as np
from rag_experiment_accelerator.llm.embeddings.sentence_transformer_embedding import SentenceTransformerEmbeddingModel

@patch("rag_experiment_accelerator.llm.embeddings.sentence_transformer_embedding.SentenceTransformer.encode")
def test_generate_embedding(mock_encode):
    expected_embeddings = np.array([0.026249676942825317, 0.013395567424595356])
    mock_encode.return_value = expected_embeddings

    model = SentenceTransformerEmbeddingModel("all-mpnet-base-v2")
    embeddings = model.generate_embedding("Hello world")

    assert embeddings == expected_embeddings.tolist()


def test_try_retrieve_model_raises_non_existing_model():
    with pytest.raises(Exception):
        SentenceTransformerEmbeddingModel("non-existing-model", 123)


def test_try_retrieve_model_raises_unsupported_model():
    with pytest.raises(Exception):
        SentenceTransformerEmbeddingModel("non-existing-model")


def test_try_retrieve_model_succeeds():
    try:
        SentenceTransformerEmbeddingModel("all-mpnet-base-v2")
    except:
        assert False, "Should not have thrown an exception"
