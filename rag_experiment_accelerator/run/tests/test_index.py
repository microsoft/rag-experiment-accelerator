from unittest.mock import MagicMock, patch
import pytest

from rag_experiment_accelerator.run.index import run


@pytest.fixture
def mock_config():
    return MagicMock()


@pytest.fixture
def mock_preprocess():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_create_acs_index():
    return MagicMock()


@pytest.fixture
def mock_load_documents():
    return MagicMock()


@pytest.fixture
def mock_cluster():
    return MagicMock()


@pytest.fixture
def mock_upload_data():
    return MagicMock()


@pytest.fixture
def mock_embedding_model():
    return MagicMock()


def test_run(
    mock_config,
    mock_embedding_model,
    mock_preprocess,
    mock_logger,
    mock_create_acs_index,
    mock_load_documents,
    mock_cluster,
    mock_upload_data,
):
    # Arrange
    config_dir = "config_dir"
    data_dir = "data_dir"
    filename = "config.json"

    mock_config.return_value = mock_config
    mock_config.artifacts_dir = "artifacts_dir"
    mock_config.CHUNK_SIZES = [10, 20]
    mock_config.OVERLAP_SIZES = [5, 10]

    # Create a list of mock EmbeddingModel instances
    embedding_models = [mock_embedding_model for _ in range(2)]

    # Set a side effect to assign a dimension value to each embedding model
    mock_embedding_model.side_effect = [
        MagicMock(dimension=100 * i) for i in range(1, 3)
    ]
    mock_config.embedding_models = embedding_models
    mock_config.EF_CONSTRUCTIONS = ["ef_construction1", "ef_construction2"]
    mock_config.EF_SEARCHES = ["ef_search1", "ef_search2"]
    mock_config.SAMPLE_DATA = True
    mock_config.SAMPLE_PERCENTAGE = 50
    mock_config.NAME_PREFIX = "prefix"
    mock_config.LANGUAGE = {"analyzers": ["analyzer1", "analyzer2"]}
    mock_config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT = (
        "service_endpoint"
    )
    mock_config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY = "admin_key"
    mock_config.AzureDocumentIntelligenceCredentials = (
        "document_intelligence_credentials"
    )
    mock_config.DATA_FORMATS = ["format1", "format2"]
    mock_config.data_dir = "data_dir"
    mock_config.CHUNKING_STRATEGY = "chunking_strategy"
    mock_config.AZURE_OAI_CHAT_DEPLOYMENT_NAME = "oai_deployment_name"

    mock_preprocess.preprocess.return_value = "preprocessed_value"

    mock_load_documents.return_value = [
        {"doc1": "value1"},
        {"doc2": "value2"},
        {"doc3": "value3"},
    ]

    mock_cluster.return_value = [
        {"cluster1": "value1"},
        {"cluster2": "value2"},
        {"cluster3": "value3"},
    ]

    # Act
    with patch("rag_experiment_accelerator.run.index.Config", mock_config), patch(
        "rag_experiment_accelerator.run.index.Preprocess", mock_preprocess
    ), patch("rag_experiment_accelerator.run.index.logger", mock_logger), patch(
        "rag_experiment_accelerator.run.index.create_acs_index", mock_create_acs_index
    ), patch(
        "rag_experiment_accelerator.run.index.load_documents", mock_load_documents
    ), patch(
        "rag_experiment_accelerator.run.index.cluster", mock_cluster
    ), patch(
        "rag_experiment_accelerator.run.index.upload_data", mock_upload_data
    ), patch(
        "rag_experiment_accelerator.run.index.open", create=True
    ), patch(
        "rag_experiment_accelerator.run.index.json.dump"
    ) as mock_json_dump:
        run(config_dir, data_dir, filename)

        # Assert
        assert mock_config.call_count == 1
        assert mock_preprocess.call_count == 1
        assert mock_logger.error.call_count == 0
        assert mock_create_acs_index.call_count == 32
        assert mock_load_documents.call_count == 4
        assert mock_cluster.call_count == 4
        assert mock_upload_data.call_count == 32
        assert mock_json_dump.call_count == 1
        assert mock_logger.info.call_count == 32
        assert mock_create_acs_index.call_args_list[0][0][0] == "service_endpoint"
        assert mock_create_acs_index.call_args_list[0][0][2] == "admin_key"

        assert mock_load_documents.call_args_list[0][0][0] == "chunking_strategy"
        assert (
            mock_load_documents.call_args_list[0][0][1]
            == "document_intelligence_credentials"
        )
        assert mock_load_documents.call_args_list[0][0][2] == ["format1", "format2"]
        assert mock_load_documents.call_args_list[0][0][3] == "data_dir"
        assert mock_load_documents.call_args_list[0][0][4] == 10
        assert mock_load_documents.call_args_list[0][0][5] == 5
        # ... assert other load_documents calls

        assert mock_cluster.call_args_list[0][0][0] == [
            {"doc1": "value1"},
            {"doc2": "value2"},
            {"doc3": "value3"},
        ]
        assert mock_cluster.call_args_list[0][0][1] == "data_dir"
        assert mock_cluster.call_args_list[0][0][2] == mock_config
