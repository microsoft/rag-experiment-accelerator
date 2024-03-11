from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.run.qa_generation import run


@patch("rag_experiment_accelerator.run.qa_generation.Config")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.exists")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.cluster")
@patch("rag_experiment_accelerator.run.qa_generation.os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
def test_run(
    mock_create_data_asset,
    mock_generate_qna,
    mock_makedirs,
    mock_cluster,
    mock_load_documents,
    mock_exists,
    mock_get_default_az_cred,
    mock_config,
):
    # Arrange
    config_dir = "test_config_dir"
    data_dir = "test_data_dir"
    filename = "test_config.json"
    config_instance = MagicMock()
    azure_cred_instance = MagicMock()
    df_instance = MagicMock()

    mock_config.return_value = config_instance
    mock_config.SAMPLE_DATA = False
    mock_get_default_az_cred.return_value = azure_cred_instance
    mock_exists.return_value = False
    mock_load_documents.return_value = all_docs_instance = MagicMock()
    mock_cluster.return_value = all_docs_instance = MagicMock()
    mock_makedirs.side_effect = None
    mock_generate_qna.return_value = df_instance

    # Act
    run(config_dir, data_dir, filename)

    # Assert
    mock_config.assert_called_once_with(config_dir, filename=filename)
    mock_get_default_az_cred.assert_called_once()
    mock_exists.assert_called_once_with(
        f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{config_instance.SAMPLE_OPTIMUM_K}.csv"
    )
    mock_load_documents.assert_called_once_with(
        config_instance.CHUNKING_STRATEGY,
        config_instance.AzureDocumentIntelligenceCredentials,
        config_instance.DATA_FORMATS,
        config_instance.data_dir,
        2000,
        0,
    )

    mock_makedirs.assert_called_once_with(config_instance.artifacts_dir, exist_ok=True)
    mock_generate_qna.assert_called_once_with(
        all_docs_instance, config_instance.AZURE_OAI_CHAT_DEPLOYMENT_NAME
    )
    df_instance.to_json.assert_called_once_with(
        config_instance.EVAL_DATA_JSONL_FILE_PATH, orient="records", lines=True
    )
    mock_create_data_asset.assert_called_once_with(
        config_instance.EVAL_DATA_JSONL_FILE_PATH,
        "eval_data",
        azure_cred_instance,
        config_instance.AzureMLCredentials,
    )
