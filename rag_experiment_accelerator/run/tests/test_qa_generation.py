import pandas as pd
from unittest.mock import patch, MagicMock
from rag_experiment_accelerator.run.qa_generation import run
import pytest


@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
@patch("rag_experiment_accelerator.run.qa_generation.exists")
@patch("rag_experiment_accelerator.run.qa_generation.pd.read_csv")
@patch("rag_experiment_accelerator.sampling.clustering.chunk_dict_to_dataframe")
def test_run_success(
    mock_config,
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
    mock_generate_qna,
    mock_create_data_asset,
    mock_read_csv,
    mock_exists,
):
    # Arrange
    data_dir = "test_data_dir"
    mock_config.SAMPLE_DATA = False
    mock_get_default_az_cred.return_value = "test_cred"
    mock_df = MagicMock()
    mock_generate_qna.return_value = mock_df
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame()

    # Act
    run("test_dir")

    # Assert
    mock_makedirs.assert_called_once()
    mock_config.assert_called_once_with("test_dir", filename="config.json")
    mock_get_default_az_cred.assert_called_once()
    mock_load_documents.assert_called_once()
    mock_generate_qna.assert_called_once()
    mock_df.to_json.assert_called_once()
    mock_create_data_asset.assert_called_once()
    mock_exists.assert_called_once_with(
        f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{mock_config.SAMPLE_OPTIMUM_K}.csv"
    )
    mock_read_csv.assert_called_once_with(
        f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{mock_config.SAMPLE_OPTIMUM_K}.csv"
    )


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
    mock_makedirs.side_effect = Exception("Unable to create the ")

    # Act and Assert
    with pytest.raises(Exception):
        run("test_dir")
