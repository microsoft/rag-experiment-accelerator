import json
import os
from rag_experiment_accelerator.config.config import Config
from unittest.mock import patch


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
def test_config_init():
    # Load mock config data from a YAML file
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    config = Config(get_test_config_dir())

    assert config.NAME_PREFIX == mock_config_data["name_prefix"]
    assert config.CHUNK_SIZES == mock_config_data["chunking"]["chunk_size"]
    assert config.OVERLAP_SIZES == mock_config_data["chunking"]["overlap_size"]
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

    st_embedding_model = config.embedding_models[0]
    assert st_embedding_model.name == "all-MiniLM-L6-v2"
    assert st_embedding_model.dimension == 384

    aoai_embedding_model = config.embedding_models[1]
    assert aoai_embedding_model.name == "text-embedding-ada-002"
    assert aoai_embedding_model.dimension == 1536


# validation tests for configuration parameters
def test_maximum_chunk_limit():
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    mock_config_data["chunking"]["chunk_size"] = 1000
    mock_config_data["chunking"]["overlap_size"] = 1000

    with open(f"{get_test_config_dir()}/config.json", "w") as file:
        json.dump(mock_config_data, file)

    config = Config(get_test_config_dir())

    assert config.CHUNK_SIZES == [512]
    assert config.OVERLAP_SIZES == [128]


def test_chunk_size_greater_than_overlap_size():
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    mock_config_data["chunking"]["chunk_size"] = 128
    mock_config_data["chunking"]["overlap_size"] = 512

    with open(f"{get_test_config_dir()}/config.json", "w") as file:
        json.dump(mock_config_data, file)

    config = Config(get_test_config_dir())

    assert config.CHUNK_SIZES == [128]
    assert config.OVERLAP_SIZES == [128]


def test_chunk_size_less_than_overlap_size():
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    mock_config_data["chunking"]["chunk_size"] = 128
    mock_config_data["chunking"]["overlap_size"] = 64

    with open(f"{get_test_config_dir()}/config.json", "w") as file:
        json.dump(mock_config_data, file)

    config = Config(get_test_config_dir())

    assert config.CHUNK_SIZES == [128]
    assert config.OVERLAP_SIZES == [64]


# efConstruction must be between 100 and 1000
def test_validate_efConstruction():
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    mock_config_data["ef_construction"] = 10000

    with open(f"{get_test_config_dir()}/config.json", "w") as file:
        json.dump(mock_config_data, file)

    config = Config(get_test_config_dir())

    assert config.EF_CONSTRUCTIONS == [10000]


# efSearch must be between 100 and 1000
def test_validate_efSearch():
    with open(f"{get_test_config_dir()}/config.json", "r") as file:
        mock_config_data = json.load(file)

    mock_config_data["ef_search"] = 10000

    with open(f"{get_test_config_dir()}/config.json", "w") as file:
        json.dump(mock_config_data, file)

    config = Config(get_test_config_dir())

    assert config.EF_SEARCHES == [10000]
