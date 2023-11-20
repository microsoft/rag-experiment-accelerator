
import pytest
from unittest.mock import patch

from rag_experiment_accelerator.llm.embeddings.openai_embeddings import OpenAIEmbeddingsModel
from rag_experiment_accelerator.config.auth import OpenAICredentials

@patch("rag_experiment_accelerator.llm.openai_model.openai.Model.retrieve")
def test_try_retrieve_model_does_not_raise(retrieve_model_mock):
    retrieve_model_mock.return_value = {
        "capabilities": {
            "inference": True,
            "embeddings": True,
        },
        "status": "succeeded",
    }

    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    try:
        model.try_retrieve_model()
    except:
        assert False, "Should not have thrown an exception"

        
@patch("rag_experiment_accelerator.llm.openai_model.openai.Model.retrieve")
def test_try_retrieve_model_does_not_raise_openai(retrieve_model_mock):
    retrieve_model_mock.return_value ={
        "id": "davinci-msft",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai"
    }

    creds = OpenAICredentials("open_ai", "openai_api_key", None, None)
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    try:
        model.try_retrieve_model()
    except:
        assert False, "Should not have thrown an exception"


@patch("rag_experiment_accelerator.llm.openai_model.openai.Model.retrieve")
def test_embedding_try_retrieve_model_raises_no_embedding_capability(retrieve_model_mock):
    retrieve_model_mock.return_value = {
        "capabilities": {
            "inference": True,
            "embeddings": False,
        },
        "status": "succeeded",
    }

    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    with pytest.raises(ValueError):
        model.try_retrieve_model()


@patch("openai.Model.retrieve")
def test_embedding_try_retrieve_model_raises_no_inference_capability(retrieve_model_mock):
    retrieve_model_mock.return_value = {
        "capabilities": {
            "inference": False,
            "embeddings": True,
        },
        "status": "succeeded",
    }

    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    with pytest.raises(ValueError):
        model.try_retrieve_model()


@patch("rag_experiment_accelerator.llm.openai_model.openai.Model.retrieve")
def test_embedding_try_retrieve_model_raises_status_not_succeeded(retrieve_model_mock):
    retrieve_model_mock.return_value = {
        "capabilities": {
            "inference": True,
            "embeddings": True,
        },
        "status": "bad_status",
    }

    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    with pytest.raises(ValueError):
        model.try_retrieve_model()


@patch("openai.Model.retrieve")
def test_embedding_try_retrieve_model_raises_when_no_capability(retrieve_model_mock):
    retrieve_model_mock.return_value = {
        "capabilities": {
            "inference": False,
            "embeddings": False,
        },
        "status": "succeeded",
    }

    creds = OpenAICredentials("azure", "openai_api_key", "openai_api_version", "openai_endpoint")
    model = OpenAIEmbeddingsModel("text-embedding-ada-002", creds)
    with pytest.raises(ValueError):
        model.try_retrieve_model()
