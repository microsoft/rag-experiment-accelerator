from unittest.mock import patch, MagicMock
from rag_experiment_accelerator.run.qa_generation import run
import pytest


@patch("rag_experiment_accelerator.run.qa_generation.Environment")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
def test_run_success(
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
    mock_generate_qna,
    mock_create_data_asset,
    mock_config,
    mock_environment,
):
    # Arrange
    mock_get_default_az_cred.return_value = "test_cred"
    mock_df = MagicMock()
    mock_generate_qna.return_value = mock_df

    # Act
    run(mock_environment, mock_config, ["file_path_one", "file_path_two"])

    # Assert
    mock_makedirs.assert_called_once()
    mock_config.assert_called_once_with("test_dir", filename="config.json")
    mock_load_documents.assert_called_once()
    mock_generate_qna.assert_called_once()
    mock_df.to_json.assert_called_once()
    mock_create_data_asset.assert_called_once()


@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
def test_run_makedirs_exception(
    mock_config,
    mock_load_documents,
    mock_makedirs,
):
    # Arrange
    mock_makedirs.side_effect = Exception("Unable to create the ")

    # Act and Assert
    with pytest.raises(Exception):
        run("test_dir")
