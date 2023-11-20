import pytest
from rag_experiment_accelerator.llm.embeddings.sentence_transformer_emebedding import SentenceTransformerEmbeddingModel


def test_generate_embedding():    
    # only testing that the first two values are correct, because there are a lot of values
    expected_embeddings = [0.026249676942825317, 0.013395567424595356]

    model = SentenceTransformerEmbeddingModel("all-mpnet-base-v2")
    embeddings = model.generate_embedding("Hello world")
    print(embeddings)
    assert embeddings[0][0] == expected_embeddings[0]
    assert embeddings[0][1] == expected_embeddings[1]


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
