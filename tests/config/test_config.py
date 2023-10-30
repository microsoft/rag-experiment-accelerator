import pytest
import json
from rag_experiment_accelerator.config.config import (
    AzureSearchCredentials,
    AzureMLCredentials,
    OpenAICredentials,
    Config,
)
from unittest.mock import patch, MagicMock, PropertyMock


def test_init_search_credentials():
    creds = AzureSearchCredentials(
        azure_search_service_endpoint="http://example.com",
        azure_search_admin_key="somekey",
    )
    assert creds.AZURE_SEARCH_SERVICE_ENDPOINT == "http://example.com"
    assert creds.AZURE_SEARCH_ADMIN_KEY == "somekey"


@patch("rag_experiment_accelerator.config.config._get_env_var")
def test_from_env_search_credentials(mock_get_env_var):
    mock_get_env_var.side_effect = ["http://fromenv.com", "envkey"]

    creds = AzureSearchCredentials.from_env()

    assert creds.AZURE_SEARCH_SERVICE_ENDPOINT == "http://fromenv.com"
    assert creds.AZURE_SEARCH_ADMIN_KEY == "envkey"


def test_init_ml_credentials():
    creds = AzureMLCredentials(
        subscription_id="some-sub-id",
        workspace_name="some-workspace",
        resource_group_name="some-resource-group",
    )
    assert creds.SUBSCRIPTION_ID == "some-sub-id"
    assert creds.WORKSPACE_NAME == "some-workspace"
    assert creds.RESOURCE_GROUP_NAME == "some-resource-group"


@patch("rag_experiment_accelerator.config.config._get_env_var")
def test_from_env_ml_credentials(mock_get_env_var):
    mock_get_env_var.side_effect = [
        "some-sub-id-env",
        "some-workspace-env",
        "some-resource-group-env",
    ]

    creds = AzureMLCredentials.from_env()

    assert creds.SUBSCRIPTION_ID == "some-sub-id-env"
    assert creds.WORKSPACE_NAME == "some-workspace-env"
    assert creds.RESOURCE_GROUP_NAME == "some-resource-group-env"


def test_init_openai_credentials():
    creds = OpenAICredentials(
        openai_api_type="azure",
        openai_api_key="somekey",
        openai_api_version="v1",
        openai_endpoint="http://example.com",
    )
    assert creds.OPENAI_API_TYPE == "azure"
    assert creds.OPENAI_API_KEY == "somekey"
    assert creds.OPENAI_API_VERSION == "v1"
    assert creds.OPENAI_ENDPOINT == "http://example.com"


def test_init_invalid_api_type_openai_credentials():
    with pytest.raises(ValueError):
        OpenAICredentials(
            openai_api_type="invalid",
            openai_api_key="somekey",
            openai_api_version="v1",
            openai_endpoint="http://example.com",
        )


@patch("rag_experiment_accelerator.config.config._get_env_var")
def test_from_env_openai_credentials(mock_get_env_var):
    mock_get_env_var.side_effect = ["azure", "envkey", "v1", "http://envexample.com"]

    creds = OpenAICredentials.from_env()

    assert creds.OPENAI_API_TYPE == "azure"
    assert creds.OPENAI_API_KEY == "envkey"
    assert creds.OPENAI_API_VERSION == "v1"
    assert creds.OPENAI_ENDPOINT == "http://envexample.com"


@pytest.mark.parametrize(
    "api_type, expect_api_version, expect_api_base",
    [
        ("azure", "expected_version", "expected_endpoint"),
        ("open_ai", None, None),
        (None, None, None),
    ],
)
@patch(
    "rag_experiment_accelerator.config.config.openai"
)  # replace 'your_module' with the actual module name
def test_set_credentials(mock_openai, api_type, expect_api_version, expect_api_base):
    creds = OpenAICredentials(
        openai_api_type=api_type,
        openai_api_key="somekey",
        openai_api_version="expected_version",
        openai_endpoint="expected_endpoint",
    )

    creds._set_credentials()

    if api_type is not None:
        assert str(mock_openai.api_type) == api_type
        assert str(mock_openai.api_key) == "somekey"

    if api_type == "azure":
        assert str(mock_openai.api_version) == expect_api_version
        assert str(mock_openai.api_base) == expect_api_base


# Load mock config data from a YAML file
with open("tests/config/data/test_config.json", "r") as file:
    mock_config_data = json.load(file)


def mock_get_env_var(var_name: str, critical: bool, mask: bool) -> str:
    if var_name == "AZURE_SEARCH_SERVICE_ENDPOINT":
        return "test_search_endpoint"
    elif var_name == "AZURE_SEARCH_ADMIN_KEY":
        return "test_admin_key"
    elif var_name == "AZURE_SUBSCRIPTION_ID":
        return "test_subscription_id"
    elif var_name == "AZURE_WORKSPACE_NAME":
        return "test_workspace_name"
    elif var_name == "AZURE_RESOURCE_GROUP_NAME":
        return "test_resource_group_name"
    elif var_name == "OPENAI_API_KEY":
        return "test_api_key"
    elif var_name == "OPENAI_API_VERSION":
        return "test_api_version"
    elif var_name == "OPENAI_API_VERSION":
        return "test_api_version"
    elif var_name == "OPENAI_API_TYPE":
        return "azure"


@patch("rag_experiment_accelerator.config.config.openai.Model.retrieve")
@patch("rag_experiment_accelerator.config.config._get_env_var", new=mock_get_env_var)
def test_config_init(
    mock_openai_model_retrieve,
):
    mock_openai_model_retrieve.return_value = {
        "status": "succeeded",
        "capabilities": {
            "embeddings": True,
            "inference": True,
            "chat_completion": True,
        }
    }
    config = Config("tests/config/data/test_config.json")
    assert config.NAME_PREFIX == mock_config_data["name_prefix"]
    assert config.CHUNK_SIZES == mock_config_data["chunking"]["chunk_size"]
    assert config.OVERLAP_SIZES == mock_config_data["chunking"]["overlap_size"]
    assert config.EMBEDDING_DIMENSIONS == mock_config_data["embedding_dimension"]
    assert config.EF_CONSTRUCTIONS == mock_config_data["efConstruction"]
    assert config.EF_SEARCHES == mock_config_data["efSearch"]
    assert config.RERANK == mock_config_data["rerank"]
    assert config.RERANK_TYPE == mock_config_data["rerank_type"]
    assert config.LLM_RERANK_THRESHOLD == mock_config_data["llm_re_rank_threshold"]
    assert config.CROSSENCODER_AT_K == mock_config_data["cross_encoder_at_k"]
    assert config.CROSSENCODER_MODEL == mock_config_data["crossencoder_model"]
    assert config.SEARCH_VARIANTS == mock_config_data["search_types"]
    assert config.METRIC_TYPES == mock_config_data["metric_types"]
    assert config.CHAT_MODEL_NAME == mock_config_data["chat_model_name"]
    assert config.EMBEDDING_MODEL_NAME == mock_config_data["embedding_model_name"]
    assert config.TEMPERATURE == mock_config_data["openai_temperature"]
    assert config.SEARCH_RELEVANCY_THRESHOLD == mock_config_data["search_relevancy_threshold"]
    assert config.DATA_FORMATS == mock_config_data["data_formats"]
    assert mock_openai_model_retrieve.called  # Ensure that the OpenAI model is retrieved

