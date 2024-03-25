from unittest.mock import patch, MagicMock

from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

from rag_experiment_accelerator.embedding.aoai_embedding_model import AOAIEmbeddingModel


@patch(
    "rag_experiment_accelerator.embedding.aoai_embedding_model.AOAIEmbeddingModel._initilize_client"
)
def test_generate_embedding(mock_client):
    expected_embeddings = Embedding(
        embedding=[0.1, 0.2, 0.3], index=0, object="embedding"
    )
    mock_embeddings = CreateEmbeddingResponse(
        data=[expected_embeddings],
        model="model_name",
        object="list",
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )

    mock_client().embeddings.create.return_value = mock_embeddings

    environment = MagicMock()
    model = AOAIEmbeddingModel("text-embedding-ada-002", environment=environment)
    embeddings = model.generate_embedding("Hello world")
    assert embeddings == mock_embeddings.data[0].embedding


def test_emebdding_dimension_has_default():
    environment = MagicMock()
    model = AOAIEmbeddingModel("text-embedding-ada-002", environment)
    assert model.dimension == 1536


def test_can_set_embedding_dimension():
    environment = MagicMock()
    model = AOAIEmbeddingModel("text-embedding-ada-002", environment, 123)
    assert model.dimension == 123
