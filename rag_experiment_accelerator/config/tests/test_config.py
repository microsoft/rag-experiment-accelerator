from rag_experiment_accelerator.config.config import Config
from unittest.mock import MagicMock, patch

import pytest
import json
import os

import sys
sys.path.insert(0, '/Users/shivamsingh/Documents/MS/rag-experiment-accelerator')


def get_test_config_dir():
    return os.path.join(os.path.dirname(__file__), "data")


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
    "rag_experiment_accelerator.config.credentials._get_env_var",
    new=mock_get_env_var,
)
@patch(
    "rag_experiment_accelerator.config.config.EmbeddingModelFactory.create",
)
def test_config_init(mock_embedding_model_factory):
    # Load mock config data from a YAML file
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    embedding_model_1 = MagicMock()
    embedding_model_2 = MagicMock()
    embedding_model_1.name.return_value = "all-MiniLM-L6-v2"
    embedding_model_1.dimension.return_value = 384
    embedding_model_2.name.return_value = "text-embedding-ada-002"
    embedding_model_2.dimension.return_value = 1536
    mock_embedding_model_factory.side_effect = [embedding_model_1, embedding_model_2]

    config = Config(get_test_config_dir())

    config.embedding_models = [embedding_model_1, embedding_model_2]

    assert config.NAME_PREFIX == mock_config_data["name_prefix"]
    assert config.CHUNK_SIZES == mock_config_data["chunking"]["chunk_size"]
    assert config.OVERLAP_SIZES == mock_config_data["chunking"]["overlap_size"]
    assert config.CHUNKING_STRATEGY == mock_config_data["chunking_strategy"]
    assert config.EF_CONSTRUCTIONS == mock_config_data["ef_construction"]
    assert config.EF_SEARCHES == mock_config_data["ef_search"]
    assert config.RERANK == mock_config_data["rerank"]
    assert config.RERANK_TYPE == mock_config_data["rerank_type"]
    assert config.LLM_RERANK_THRESHOLD == mock_config_data["llm_re_rank_threshold"]
    assert config.CROSSENCODER_AT_K == mock_config_data["cross_encoder_at_k"]
    assert config.CROSSENCODER_MODEL == mock_config_data["crossencoder_model"]
    assert config.SEARCH_VARIANTS == mock_config_data["search_types"]
    assert config.METRIC_TYPES == mock_config_data["metric_types"]
    assert (
        config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
        == mock_config_data["azure_oai_chat_deployment_name"]
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

    assert config.embedding_models[0].name.return_value == "all-MiniLM-L6-v2"
    assert config.embedding_models[0].dimension.return_value == 384

    assert config.embedding_models[1].name.return_value == "text-embedding-ada-002"
    assert config.embedding_models[1].dimension.return_value == 1536


def test_chunk_size_greater_than_overlap_size():
    with pytest.raises(ValueError) as info:
        Config.validate_inputs(Config, [128], [512], [400], [400])

    assert (
        str(info.value)
        == "Config param validation error: overlap_size must be less than chunk_size"
    )


def test_validate_ef_search():
    with pytest.raises(ValueError) as high_info:
        Config.validate_inputs(Config, [512], [128], [400], [1001])

    with pytest.raises(ValueError) as low_info:
        Config.validate_inputs(Config, [512], [128], [400], [99])

    assert (
        str(high_info.value)
        == "Config param validation error: ef_search must be between 100 and 1000 (inclusive)"
    )
    assert (
        str(low_info.value)
        == "Config param validation error: ef_search must be between 100 and 1000 (inclusive)"
    )


def test_validate_ef_construction():
    with pytest.raises(ValueError) as high_info:
        Config.validate_inputs(Config, [512], [128], [1001], [400])

    with pytest.raises(ValueError) as low_info:
        Config.validate_inputs(Config, [512], [128], [99], [400])

    assert (
        str(high_info.value)
        == "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
    )
    assert (
        str(low_info.value)
        == "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
    )
