from unittest.mock import patch, MagicMock
from pandas import DataFrame
from rag_experiment_accelerator.run.qa_generation import run
from unittest import TestCase


@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
def test_run_success(
    mock_config,
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
    mock_generate_qna,
    mock_create_data_asset,
):
    # Arrange
    mock_config.return_value = MagicMock()
    mock_get_default_az_cred.return_value = "test_cred"
    mock_load_documents.return_value = MagicMock()
    mock_makedirs.side_effect = None
    mock_df = MagicMock(DataFrame)
    mock_generate_qna.return_value = mock_df
    mock_create_data_asset.side_effect = None

    # Act
    run("test_dir")

    # Assert
    mock_makedirs.assert_called_once()
    mock_config.assert_called_once_with("test_dir")
    mock_get_default_az_cred.assert_called_once()
    mock_load_documents.assert_called_once()
    mock_generate_qna.assert_called_once()
    mock_df.to_json.assert_called_once()
    mock_create_data_asset.assert_called_once()


@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
def test_run_makedirs_exception(
    mock_config,
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
):
    # Arrange
    mock_config.return_value = MagicMock()
    mock_get_default_az_cred.return_value = "test_cred"
    mock_load_documents.return_value = MagicMock()
    mock_makedirs.side_effect = Exception("Unable to create the ")
    tc = TestCase()

    # Act and Assert
    with tc.assertRaises(Exception) as context:
        run("test_dir")
    tc.assertTrue("Unable to create the " in str(context.exception))
