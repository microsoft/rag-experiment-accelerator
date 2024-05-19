from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.checkpoint.checkpoint import init_checkpoint
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_file_paths


@patch("rag_experiment_accelerator.run.index.mlflow")
@patch("rag_experiment_accelerator.run.index.mlflow.MlflowClient")
@patch("rag_experiment_accelerator.embedding.embedding_model.EmbeddingModel")
@patch("rag_experiment_accelerator.run.index.upload_data")
@patch("rag_experiment_accelerator.run.index.cluster")
@patch("rag_experiment_accelerator.run.index.load_documents")
@patch("rag_experiment_accelerator.run.index.create_acs_index")
@patch("rag_experiment_accelerator.run.index.Preprocess")
@patch("rag_experiment_accelerator.run.index.Config.__init__", return_value=None)
@patch("rag_experiment_accelerator.run.index.Environment")
def test_run(
    mock_environment,
    _,
    mock_preprocess,
    mock_create_acs_index,
    mock_load_documents,
    mock_cluster,
    mock_upload_data,
    mock_embedding_model,
    mock_mlflow_client,
    __,
):
    # Arrange
    data_dir = "./data"

    mock_config = Config()
    mock_config.artifacts_dir = "artifacts_dir"
    mock_config.preprocess = False
    mock_config.chunk_sizes = [10, 20]
    mock_config.overlap_sizes = [5, 10]

    # Create a list of mock EmbeddingModel instances
    embedding_models = [mock_embedding_model for _ in range(2)]

    # Set a side effect to assign a dimension value to each embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.side_effect = [
        MagicMock(dimension=100 * i) for i in range(1, 3)
    ]

    mock_config.embedding_models = embedding_models
    mock_config.ef_constructions = ["ef_construction1", "ef_construction2"]
    mock_config.ef_searches = ["ef_search1", "ef_search2"]
    mock_config.data_formats = "test_format"
    mock_config.chunking_strategy = "basic"
    mock_config.max_worker_threads = 1
    mock_config.sampling = False
    mock_config.index_name_prefix = "prefix"
    mock_config.language = {"analyzers": ["analyzer1", "analyzer2"]}
    mock_config.generate_title = False
    mock_config.generate_summary = False
    mock_config.override_content_with_summary = False
    mock_config.azure_document_intelligence_model = "prebuilt-read"
    mock_config.data_formats = ["format1", "format2"]
    mock_config.data_dir = "data_dir"
    mock_config.use_checkpoints = False
    mock_config.chunking_strategy = "chunking_strategy"
    mock_config.azure_oai_chat_deployment_name = "oai_deployment_name"

    mock_environment.azure_search_service_endpoint = "service_endpoint"
    mock_environment.azure_search_admin_key = "admin_key"
    mock_environment.azure_document_intelligence_endpoint = (
        "document_intelligence_endpoint"
    )
    mock_environment.azure_document_intelligence_key = "document_intelligence_key"

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
        init_checkpoint(mock_config)
        run(mock_environment, mock_config, index_config, file_paths, mock_mlflow_client)

    # Assert
    assert mock_preprocess.call_count == 32
    assert mock_create_acs_index.call_count == 32
    assert mock_load_documents.call_count == 32
    assert mock_upload_data.call_count == 32
    assert mock_create_acs_index.call_args_list[0][0][0] == "service_endpoint"
    assert mock_create_acs_index.call_args_list[0][0][2] == "admin_key"

    assert mock_load_documents.call_args_list[0][0][1] == "chunking_strategy"
    assert mock_load_documents.call_args_list[0][0][2] == ["format1", "format2"]
    assert mock_load_documents.call_args_list[0][0][3] == file_paths
    assert mock_load_documents.call_args_list[0][0][4] == 10
    assert mock_load_documents.call_args_list[0][0][5] == 5
