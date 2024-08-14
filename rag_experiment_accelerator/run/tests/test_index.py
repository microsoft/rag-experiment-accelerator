from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.checkpoint.checkpoint_factory import init_checkpoint
from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.language_config import LanguageConfig
from rag_experiment_accelerator.config.openai_config import OpenAIConfig
from rag_experiment_accelerator.config.query_expansion import QueryExpansionConfig
from rag_experiment_accelerator.config.sampling_config import SamplingConfig
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_file_paths


@patch("rag_experiment_accelerator.run.index.mlflow.MlflowClient")
@patch("rag_experiment_accelerator.config.config.create_embedding_model")
@patch("rag_experiment_accelerator.run.index.upload_data")
@patch("rag_experiment_accelerator.run.index.cluster")
@patch("rag_experiment_accelerator.run.index.load_documents")
@patch("rag_experiment_accelerator.run.index.create_acs_index")
@patch("rag_experiment_accelerator.run.index.Preprocess")
@patch("rag_experiment_accelerator.run.index.Environment")
def test_run(
    mock_environment,
    mock_preprocess,
    mock_create_acs_index,
    mock_load_documents,
    mock_cluster,
    mock_upload_data,
    mock_create_embedding_model,
    mock_mlflow_client,
):
    # Arrange
    data_dir = "./data"

    embedding_model_1 = MagicMock()
    embedding_model_1.deployment_name.return_value = "all-MiniLM-L6-v2"
    embedding_model_1.dimension.return_value = 384

    embedding_model_2 = MagicMock()
    embedding_model_2.deployment_name.return_value = "text-embedding-ada-002"
    embedding_model_2.dimension.return_value = 1536
    mock_create_embedding_model.side_effect = [embedding_model_1, embedding_model_2]

    mock_config = MagicMock(spec=Config)
    mock_config.data_formats = ["format1", "format2"]
    mock_config.use_checkpoints = False
    mock_config.max_worker_threads = 1

    mock_config.use_checkpoints = False

    mock_config.index = IndexConfig(
        index_name_prefix="prefix",
        ef_construction=[300, 400],
        ef_search=[300, 400],
        chunking=ChunkingConfig(
            chunk_size=[10, 20],
            overlap_size=[5, 10],
            chunking_strategy="chunking_strategy",
            generate_title=False,
            generate_summary=False,
            override_content_with_summary=False,
            azure_document_intelligence_model="prebuilt-read",
        ),
        embedding_model=[MagicMock(), MagicMock()],
    )

    mock_config.sampling = MagicMock(spec=SamplingConfig, sample_data=False)

    mock_config.language = MagicMock(
        spec=LanguageConfig, analyzer=["analyzer1", "analyzer2"]
    )

    mock_config.query_expansion = MagicMock(
        spec=QueryExpansionConfig, query_expansion=False
    )

    mock_config.openai = MagicMock(spec=OpenAIConfig)
    mock_config.openai.azure_oai_chat_deployment_name = "test-deployment"

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
    for index_config in mock_config.index.flatten():
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
