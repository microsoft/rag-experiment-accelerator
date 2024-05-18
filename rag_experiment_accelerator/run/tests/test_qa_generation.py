from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.run.qa_generation import run


@patch("rag_experiment_accelerator.run.qa_generation.Environment")
@patch("rag_experiment_accelerator.run.qa_generation.exists")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.cluster")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
def test_run(
    mock_create_data_asset,
    mock_generate_qna,
    mock_cluster,
    mock_load_documents,
    mock_exists,
    mock_environment,
):
    # Arrange
    data_dir = "test_data_dir"
    df_instance = MagicMock()

    mock_config = MagicMock()
    mock_config.SAMPLE_DATA = True
    mock_config.SAMPLE_OPTIMUM_K = 3
    sampled_input_data_csv_path = f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{mock_config.SAMPLE_OPTIMUM_K}.csv"
    mock_config._sample_input_data_csv_path.return_value = sampled_input_data_csv_path
    mock_exists.return_value = False

    mock_load_documents.return_value = all_docs_instance = MagicMock()
    mock_cluster.return_value = all_docs_instance = MagicMock()
    mock_generate_qna.return_value = df_instance
    filepaths = ["file_path_one", "file_path_two"]

    # Act
    run(mock_environment, mock_config, filepaths)

    # Assert
    mock_load_documents.assert_called_once_with(
        mock_environment,
        mock_config.chunking_strategy,
        mock_config.data_formats,
        filepaths,
        2000,
        0,
    )
    mock_generate_qna.assert_called_once_with(
        mock_environment,
        mock_config,
        all_docs_instance,
        mock_config.azure_oai_chat_deployment_name,
    )
    df_instance.to_json.assert_called_once_with(
        mock_config.eval_data_jsonl_file_path, orient="records", lines=True
    )
    mock_create_data_asset.assert_called_once_with(
        mock_config.eval_data_jsonl_file_path, "eval_data", mock_environment
    )
