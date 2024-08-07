import pytest
import json
import os
from rag_experiment_accelerator.config.config import Config
from unittest.mock import MagicMock, patch


class ConfigNoInit(Config):
    """
    This class represents a configuration object without an initializer.

    It inherits from the `Config` class.

    Usage:
    ```
    config = ConfigNoInit()
    ```
    """

    def __init__(self) -> None:
        pass


def get_test_config_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@patch(
    "rag_experiment_accelerator.config.config.create_embedding_model",
)
def test_config_init(mock_create_embedding_model):
    # Load mock config data from a YAML file
    config_path = f"{get_test_config_dir()}/config.json"
    with open(config_path, "r") as file:
        mock_config_data = json.load(file)

    embedding_model_1 = MagicMock()
    embedding_model_2 = MagicMock()
    environment = MagicMock()
    embedding_model_1.name.return_value = "all-MiniLM-L6-v2"
    embedding_model_1.dimension.return_value = 384
    embedding_model_2.name.return_value = "text-embedding-ada-002"
    embedding_model_2.dimension.return_value = 1536
    mock_create_embedding_model.side_effect = [embedding_model_1, embedding_model_2]

    config = Config.from_path(environment, config_path)

    config.embedding_model = [embedding_model_1, embedding_model_2]

    assert config.index_name_prefix == mock_config_data["index_name_prefix"]
    assert config.experiment_name == mock_config_data["experiment_name"]
    assert config.chunk_sizes == mock_config_data["chunking"]["chunk_size"]
    assert config.overlap_sizes == mock_config_data["chunking"]["overlap_size"]
    assert config.chunking_strategy == mock_config_data["chunking_strategy"]
    assert config.ef_constructions == mock_config_data["ef_construction"]
    assert config.ef_searches == mock_config_data["ef_search"]
    assert config.rerank == mock_config_data["rerank"]
    assert config.rerank_type == mock_config_data["rerank_type"]
    assert config.llm_rerank_threshold == mock_config_data["llm_rerank_threshold"]
    assert config.crossencoder_at_k == mock_config_data["cross_encoder_at_k"]
    assert config.crossencoder_model == mock_config_data["crossencoder_model"]
    assert config.search_types == mock_config_data["search_types"]
    assert config.metric_types == mock_config_data["metric_types"]
    assert (
        config.azure_oai_chat_deployment_name
        == mock_config_data["azure_oai_chat_deployment_name"]
    )
    assert config.temperature == mock_config_data["openai_temperature"]
    assert (
        config.search_relevency_threshold
        == mock_config_data["search_relevancy_threshold"]
    )
    assert config.data_formats == mock_config_data["data_formats"]
    assert (
        config.eval_data_jsonl_file_path
        == f"{get_test_config_dir()}/artifacts/eval_data.jsonl"
    )

    assert config.embedding_model[0].name.return_value == "all-MiniLM-L6-v2"
    assert config.embedding_model[0].dimension.return_value == 384

    assert config.embedding_model[1].name.return_value == "text-embedding-ada-002"
    assert config.embedding_model[1].dimension.return_value == 1536

    assert config.sampling
    assert config.sample_percentage == mock_config_data["sampling"]["sample_percentage"]
    assert config.sample_optimum_k == mock_config_data["sampling"]["optimum_k"]
    assert config.sample_min_cluster == mock_config_data["sampling"]["min_cluster"]
    assert config.sample_max_cluster == mock_config_data["sampling"]["max_cluster"]


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


def test_validate_semantic_search_config():
    config = ConfigNoInit()

    # Test case 1: use_semantic_search is False, but semantic search is
    # required
    config.search.search_type = ["search_for_match_semantic"]
    use_semantic_search = False
    with pytest.raises(ValueError) as info:
        config.validate_semantic_search_config(use_semantic_search)
    assert (
        str(info.value)
        == "Semantic search is required for search types 'search_for_match_semantic' or 'search_for_manual_hybrid', but it's not enabled."
    )

    # Test case 2: use_semantic_search is True, and semantic search is required
    config.search.search_type = ["search_for_match_semantic"]
    use_semantic_search = True
    # No exception should be raised
    config.validate_semantic_search_config(use_semantic_search)

    # Test case 3: use_semantic_search is False, and semantic search is not
    # required
    config.search.search_type = ["search_for_exact_match"]
    use_semantic_search = False
    # No exception should be raised
    config.validate_semantic_search_config(use_semantic_search)

    # Test case 4: use_semantic_search is True, and semantic search is not
    # required
    config.search.search_type = ["search_for_exact_match"]
    use_semantic_search = True
    # No exception should be raised
    config.validate_semantic_search_config(use_semantic_search)
