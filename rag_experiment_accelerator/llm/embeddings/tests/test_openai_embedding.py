from unittest.mock import patch

from rag_experiment_accelerator.llm.embeddings.openai_embedding import OpenAIEmbeddingModel
from rag_experiment_accelerator.config.auth import OpenAICredentials

@patch("rag_experiment_accelerator.llm.embeddings.openai_embedding.openai.Embedding.create")
def test_generate_embedding(create_embedding_mock):
    expected_embeddings = [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ]
    create_embedding_mock.return_value = {"data": [{"embedding": expected_embeddings}]}

    creds = OpenAICredentials("open_ai", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingModel("text-embedding-ada-002", creds)
    embeddings = model.generate_embedding("Hello world")
    assert embeddings == [expected_embeddings]

 
def test_emebdding_dimension_has_default():
    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingModel("text-embedding-ada-002", creds)
    assert model.dimension == 1536


def test_can_set_embedding_dimension():
    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingModel("text-embedding-ada-002", creds, 123)
    assert model.dimension == 123

