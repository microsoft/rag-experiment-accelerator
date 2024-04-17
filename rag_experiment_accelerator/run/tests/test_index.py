from unittest.mock import MagicMock, patch
from rag_experiment_accelerator.config.config import Config

from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_file_paths


@patch("rag_experiment_accelerator.config.config.os.makedirs")
@patch("rag_experiment_accelerator.embedding.embedding_model.EmbeddingModel")
@patch("rag_experiment_accelerator.run.index.upload_data")
@patch("rag_experiment_accelerator.run.index.cluster")
@patch("rag_experiment_accelerator.run.index.load_documents")
@patch("rag_experiment_accelerator.run.index.create_acs_index")
@patch("rag_experiment_accelerator.run.index.logger")
@patch("rag_experiment_accelerator.run.index.Preprocess")
@patch("rag_experiment_accelerator.run.index.Config.__init__", return_value=None)
@patch("rag_experiment_accelerator.run.index.Environment")
def test_run(
    mock_environment,
    _,
    mock_preprocess,
    mock_logger,
    mock_create_acs_index,
    mock_load_documents,
    mock_cluster,
    mock_upload_data,
    mock_embedding_model,
    __,
):
    # Arrange
    data_dir = "./data"

    mock_config = Config()
    mock_config.artifacts_dir = "artifacts_dir"
    mock_config.PREPROCESS = False
    mock_config.CHUNK_SIZES = [10, 20]
    mock_config.OVERLAP_SIZES = [5, 10]

    # Create a list of mock EmbeddingModel instances
    embedding_models = [mock_embedding_model for _ in range(2)]

    # Set a side effect to assign a dimension value to each embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.side_effect = [
        MagicMock(dimension=100 * i) for i in range(1, 3)
    ]
    mock_config.embedding_models = embedding_models
    mock_config.EF_CONSTRUCTIONS = ["ef_construction1", "ef_construction2"]
    mock_config.EF_SEARCHES = ["ef_search1", "ef_search2"]
    mock_config.DATA_FORMATS = "test_format"
    mock_config.CHUNKING_STRATEGY = "basic"
    mock_config.MAX_WORKER_THREADS = 1
    mock_config.SAMPLE_DATA = False
    mock_config.INDEX_NAME_PREFIX = "prefix"
    mock_config.LANGUAGE = {"analyzers": ["analyzer1", "analyzer2"]}
    mock_config.GENERATE_TITLE = False
    mock_config.GENERATE_SUMMARY = False
    mock_config.OVERRIDE_CONTENT_WITH_SUMMARY = False
    mock_config.AZURE_DOCUMENT_INTELLIGENCE_MODEL = "prebuilt-read"

    mock_environment.azure_search_service_endpoint = "service_endpoint"
    mock_environment.azure_search_admin_key = "admin_key"
    mock_environment.azure_document_intelligence_endpoint = (
        "document_intelligence_endpoint"
    )
    mock_environment.azure_document_intelligence_key = "document_intelligence_key"

    mock_config.DATA_FORMATS = ["format1", "format2"]
    mock_config.data_dir = "data_dir"
    mock_config.CHUNKING_STRATEGY = "chunking_strategy"
    mock_config.AZURE_OAI_CHAT_DEPLOYMENT_NAME = "oai_deployment_name"

    mock_preprocess.return_value.preprocess.return_value = "preprocessed_value"

    mock_load_documents.return_value = [
        {"key1": {"content": "content1", "metadata": {"source": "source1"}}},
        {"key2": {"content": "content2", "metadata": {"source": "source2"}}},
        {"key3": {"content": "content3", "metadata": {"source": "source3"}}},
    ]

    mock_cluster.return_value = [
        {"cluster1": {"content": "content1", "metadata": {"source": "source1"}}},
        {"cluster2": {"content": "content2", "metadata": {"source": "source2"}}},
        {"cluster3": {"content": "content3", "metadata": {"source": "source3"}}},
    ]
    file_paths = get_all_file_paths(data_dir)

    # Act
    for index_config in mock_config.index_configs():
        run(mock_environment, mock_config, index_config, file_paths)

    # Assert
    assert mock_preprocess.call_count == 32
    assert mock_logger.error.call_count == 0
    assert mock_create_acs_index.call_count == 32
    assert mock_load_documents.call_count == 32
    # assert mock_cluster.call_count == 32
    assert mock_upload_data.call_count == 32
    assert mock_create_acs_index.call_args_list[0][0][0] == "service_endpoint"
    assert mock_create_acs_index.call_args_list[0][0][2] == "admin_key"

    assert mock_load_documents.call_args_list[0][0][1] == "chunking_strategy"
    assert mock_load_documents.call_args_list[0][0][2] == ["format1", "format2"]
    assert mock_load_documents.call_args_list[0][0][3] == file_paths
    assert mock_load_documents.call_args_list[0][0][4] == 10
    assert mock_load_documents.call_args_list[0][0][5] == 5

    # assert mock_cluster.call_args_list[0][0][0] == [
    #     {"key1": {"content": "content1", "metadata": {"source": "source1"}}},
    #     {"key2": {"content": "content2", "metadata": {"source": "source2"}}},
    #     {"key3": {"content": "content3", "metadata": {"source": "source3"}}},
    # ]
    # assert mock_cluster.call_args_list[0][0][1] == mock_config
