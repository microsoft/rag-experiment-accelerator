import json
import os
from unittest.mock import patch

import pytest

from rag_experiment_accelerator.config.config import (AzureMLCredentials,
                                                      AzureSearchCredentials,
                                                      Config,
                                                      OpenAICredentials,
                                                      _get_env_var,
                                                      _mask_string)


def get_test_config_dir():
    return os.path.join(os.path.dirname(__file__), "data")


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


def test_raises_when_openai_endpoint_is_none_for_azure_openai():
    with pytest.raises(ValueError):
        OpenAICredentials(
            openai_api_type="azure",
            openai_api_key="somekey",
            openai_api_version="v1",
            openai_endpoint=None,
        )


def test_raises_when_openai_api_version_is_none_for_azure_openai():
    with pytest.raises(ValueError):
        OpenAICredentials(
            openai_api_type="azure",
            openai_api_key="somekey",
            openai_api_version=None,
            openai_endpoint="http://example.com",
        )


@patch("rag_experiment_accelerator.config.config._get_env_var")
def test_from_env_openai_credentials(mock_get_env_var):
    mock_get_env_var.side_effect = [
        "azure",
        "envkey",
        "v1",
        "http://envexample.com",
    ]

    creds = OpenAICredentials.from_env()

    assert creds.OPENAI_API_TYPE == "azure"
    assert creds.OPENAI_API_KEY == "envkey"
    assert creds.OPENAI_API_VERSION == "v1"
    assert creds.OPENAI_ENDPOINT == "http://envexample.com"


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
    elif var_name == "OPENAI_ENDPOINT":
        return "test_api_endpoint"
    elif var_name == "OPENAI_API_TYPE":
        return "azure"


@patch(
    "rag_experiment_accelerator.config.config._get_env_var",
    new=mock_get_env_var,
)
def test_config_init():
    # Load mock config data from a YAML file
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    config = Config(get_test_config_dir())
    assert config.NAME_PREFIX == mock_config_data["name_prefix"]
    assert config.CHUNK_SIZES == mock_config_data["chunking"]["chunk_size"]
    assert config.OVERLAP_SIZES == mock_config_data["chunking"]["overlap_size"]
    assert (
        config.EMBEDDING_DIMENSIONS == mock_config_data["embedding_dimension"]
    )
    assert config.EF_CONSTRUCTIONS == mock_config_data["ef_construction"]
    assert config.EF_SEARCHES == mock_config_data["ef_search"]
    assert config.RERANK == mock_config_data["rerank"]
    assert config.RERANK_TYPE == mock_config_data["rerank_type"]
    assert (
        config.LLM_RERANK_THRESHOLD
        == mock_config_data["llm_re_rank_threshold"]
    )
    assert config.CROSSENCODER_AT_K == mock_config_data["cross_encoder_at_k"]
    assert config.CROSSENCODER_MODEL == mock_config_data["crossencoder_model"]
    assert config.SEARCH_VARIANTS == mock_config_data["search_types"]
    assert config.METRIC_TYPES == mock_config_data["metric_types"]
    assert (
        config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
        == mock_config_data["azure_oai_chat_deployment_name"]
    )
    assert (
        config.EMBEDDING_MODEL_NAME == mock_config_data["embedding_model_name"]
    )
    assert config.TEMPERATURE == mock_config_data["openai_temperature"]
    assert (
        config.SEARCH_RELEVANCY_THRESHOLD
        == mock_config_data["search_relevancy_threshold"]
    )
    assert config.DATA_FORMATS == mock_config_data["data_formats"]
    assert (
        config.EVAL_DATA_JSONL_FILE_PATH
        == f"{get_test_config_dir()}/{mock_config_data['eval_data_jsonl_file_path']}"
    )


@pytest.mark.parametrize(
    "input_string, start, end, mask_char, expected",
    [
        ("1234567890", 2, 2, "*", "12******90"),
        ("", 2, 2, "*", ""),
        ("123", 1, 1, "*", "1*3"),
        ("1234", 2, 2, "*", "1***"),
        ("12", 1, 1, "*", "1*"),
        ("1234", 0, 0, "*", "****"),
        ("abcd", 2, 2, "#", "a###"),
    ],
)
def test_mask_string(input_string, start, end, mask_char, expected):
    result = _mask_string(input_string, start, end, mask_char)
    assert result == expected


@patch(
    "rag_experiment_accelerator.config.config.logger"
)  # Replace with the actual import path to logger
@patch("os.getenv")
@pytest.mark.parametrize(
    "var_name, critical, mask, env_value, expected_value, expected_exception,"
    " expected_log",
    [
        (
            "TEST_VAR",
            True,
            False,
            "value",
            "value",
            None,
            "TEST_VAR set to value",
        ),
        (
            "TEST_VAR",
            True,
            False,
            None,
            None,
            ValueError,
            "TEST_VAR environment variable not set.",
        ),
        (
            "TEST_VAR",
            True,
            True,
            "value",
            "value",
            None,
            "TEST_VAR set to va*ue",
        ),
    ],
)
def test_get_env_var(
    mock_getenv,
    mock_logger,
    var_name,
    critical,
    mask,
    env_value,
    expected_value,
    expected_exception,
    expected_log,
):
    mock_getenv.return_value = env_value
    if expected_exception:
        with pytest.raises(expected_exception):
            _get_env_var(var_name, critical, mask)
    else:
        assert _get_env_var(var_name, critical, mask) == expected_value
        mock_logger.info.assert_called_with(expected_log)
