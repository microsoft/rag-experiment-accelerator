import pytest
import json
import os
from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.llm.prompt import main_instruction


def init_config():
    config = Config()
    config.index.chunking.chunk_size = [512]
    config.index.chunking.overlap_size = [128]
    config.index.ef_construction = [400]
    config.index.ef_search = [400]
    return config


def get_test_config_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@patch("rag_experiment_accelerator.config.config.create_embedding_model")
def test_config_init(mock_create_embedding_model):
    # Load mock config data from a YAML file
    config_path = f"{get_test_config_dir()}/config.json"
    with open(config_path, "r") as file:
        mock_config = json.load(file)

    environment = MagicMock()
    embedding_model_1 = MagicMock()
    embedding_model_1.deployment_name.return_value = "all-MiniLM-L6-v2"
    embedding_model_1.dimension.return_value = 384
    embedding_model_2 = MagicMock()
    embedding_model_2.deployment_name.return_value = "text-embedding-ada-002"
    embedding_model_2.dimension.return_value = 1536
    mock_create_embedding_model.side_effect = [embedding_model_1, embedding_model_2]

    config = Config.from_path(environment, config_path)

    assert config.experiment_name == mock_config["experiment_name"]
    # execution_environment
    assert config.job_name == mock_config["job_name"]
    assert config.job_description == mock_config["job_description"]
    assert config.data_formats == mock_config["data_formats"]
    assert config.main_instruction.system_message == main_instruction.system_message
    assert (config.max_worker_threads is None) and (
        "max_worker_threads" not in mock_config
    )
    assert config.use_checkpoints == mock_config["use_checkpoints"]

    index = config.index
    mock_index = mock_config["index"]
    assert index.index_name_prefix == mock_index["index_name_prefix"]
    assert index.ef_construction == mock_index["ef_construction"]
    assert index.ef_search == mock_index["ef_search"]

    chunking = index.chunking
    mock_chunking = mock_config["index"]["chunking"]
    assert chunking.preprocess == mock_chunking["preprocess"]
    assert chunking.chunk_size == mock_chunking["chunk_size"]
    assert chunking.overlap_size == mock_chunking["overlap_size"]
    assert chunking.generate_title == mock_chunking["generate_title"]
    assert chunking.generate_summary == mock_chunking["generate_summary"]
    assert (
        chunking.override_content_with_summary
        == mock_chunking["override_content_with_summary"]
    )
    assert chunking.chunking_strategy == mock_chunking["chunking_strategy"]
    assert (
        chunking.azure_document_intelligence_model
        == mock_chunking["azure_document_intelligence_model"]
    )

    sampling = config.index.sampling
    assert sampling.sample_data == mock_config["index"]["sampling"]["sample_data"]
    assert sampling.percentage == mock_config["index"]["sampling"]["percentage"]
    assert sampling.optimum_k == mock_config["index"]["sampling"]["optimum_k"]
    assert sampling.min_cluster == mock_config["index"]["sampling"]["min_cluster"]
    assert sampling.max_cluster == mock_config["index"]["sampling"]["max_cluster"]

    mock_embedding = mock_config["index"]["embedding_model"]
    assert index.embedding_model[0].type == mock_embedding[0]["type"]
    assert index.embedding_model[0].model_name == mock_embedding[0]["model_name"]

    assert index.embedding_model[1].type == mock_embedding[1]["type"]
    assert index.embedding_model[1].model_name == mock_embedding[1]["model_name"]

    model1 = config.get_embedding_model(config.index.embedding_model[0].model_name)
    assert model1.deployment_name.return_value == "all-MiniLM-L6-v2"
    assert model1.dimension.return_value == 384

    model2 = config.get_embedding_model(config.index.embedding_model[1].model_name)
    assert model2.deployment_name.return_value == "text-embedding-ada-002"
    assert model2.dimension.return_value == 1536

    assert config.language.query_language == mock_config["language"]["query_language"]
    analyzer = config.language.analyzer
    mock_analyzer = mock_config["language"]["analyzer"]
    assert analyzer.analyzer_name == mock_analyzer["analyzer_name"]
    assert analyzer.index_analyzer_name == mock_analyzer["index_analyzer_name"]
    assert analyzer.search_analyzer_name == mock_analyzer["search_analyzer_name"]
    assert analyzer.char_filters == mock_analyzer["char_filters"]
    assert analyzer.tokenizers == mock_analyzer["tokenizers"]
    assert analyzer.token_filters == mock_analyzer["token_filters"]

    mock_rerank = mock_config["rerank"]
    assert config.rerank.enabled == mock_rerank["enabled"]
    assert config.rerank.type == mock_rerank["type"]
    assert config.rerank.cross_encoder_at_k == mock_rerank["cross_encoder_at_k"]
    assert config.rerank.cross_encoder_model == mock_rerank["cross_encoder_model"]
    assert config.rerank.llm_rerank_threshold == mock_rerank["llm_rerank_threshold"]

    mock_search = mock_config["search"]
    assert (
        config.search.retrieve_num_of_documents
        == mock_search["retrieve_num_of_documents"]
    )
    assert config.search.search_type == mock_search["search_type"]
    assert (
        config.search.search_relevancy_threshold
        == mock_search["search_relevancy_threshold"]
    )

    query_expansion = config.query_expansion
    mock_query_expansion = mock_config["query_expansion"]
    assert query_expansion.query_expansion == mock_query_expansion["query_expansion"]
    assert query_expansion.hyde == mock_query_expansion["hyde"]
    assert (
        query_expansion.min_query_expansion_related_question_similarity_score
        == mock_query_expansion["min_query_expansion_related_question_similarity_score"]
    )
    assert (
        query_expansion.expand_to_multiple_questions
        == mock_query_expansion["expand_to_multiple_questions"]
    )

    openai = config.openai
    mock_openai = mock_config["openai"]
    assert (
        openai.azure_oai_chat_deployment_name
        == mock_openai["azure_oai_chat_deployment_name"]
    )
    assert (
        openai.azure_oai_eval_deployment_name
        == mock_openai["azure_oai_eval_deployment_name"]
    )
    assert openai.temperature == mock_openai["temperature"]

    assert config.eval.metric_types == mock_config["eval"]["metric_types"]

    assert config.path.eval_data_file.endswith("eval_data.jsonl") and (
        "eval_data_file" not in mock_config["path"]
    )


def test_chunk_size_greater_than_overlap_size():
    config = init_config()
    config.index.chunking.chunk_size = [128]
    config.index.chunking.overlap_size = [512]

    with pytest.raises(ValueError) as info:
        config.validate_inputs()

    assert (
        str(info.value)
        == "Config param validation error: overlap_size must be less than chunk_size"
    )


def test_validate_ef_search():
    with pytest.raises(ValueError) as high_info:
        config = init_config()
        config.index.ef_search = [1001]
        config.validate_inputs()

    with pytest.raises(ValueError) as low_info:
        config = init_config()
        config.index.ef_search = [99]
        config.validate_inputs()

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
        config = init_config()
        config.index.ef_construction = [1001]
        config.validate_inputs()

    with pytest.raises(ValueError) as low_info:
        config = init_config()
        config.index.ef_construction = [99]
        config.validate_inputs()

    assert (
        str(high_info.value)
        == "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
    )
    assert (
        str(low_info.value)
        == "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
    )


def test_validate_semantic_search_config():
    config = init_config()

    # Test case 1: use_semantic_search is False, but semantic search is
    # required
    config.search.search_type = ["search_for_match_semantic"]
    use_semantic_search = False
    with pytest.raises(ValueError) as info:
        config.validate_inputs(use_semantic_search)
    assert (
        str(info.value)
        == "Semantic search is required for search types 'search_for_match_semantic' or 'search_for_manual_hybrid', but it's not enabled."
    )

    # Test case 2: use_semantic_search is True, and semantic search is required
    config.search.search_type = ["search_for_match_semantic"]
    use_semantic_search = True
    # No exception should be raised
    config.validate_inputs(use_semantic_search)

    # Test case 3: use_semantic_search is False, and semantic search is not
    # required
    config.search.search_type = ["search_for_exact_match"]
    use_semantic_search = False
    # No exception should be raised
    config.validate_inputs(use_semantic_search)

    # Test case 4: use_semantic_search is True, and semantic search is not
    # required
    config.search.search_type = ["search_for_exact_match"]
    use_semantic_search = True
    # No exception should be raised
    config.validate_inputs(use_semantic_search)
