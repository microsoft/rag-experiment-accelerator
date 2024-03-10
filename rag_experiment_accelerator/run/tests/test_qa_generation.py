from unittest.mock import patch, MagicMock
from rag_experiment_accelerator.run.qa_generation import run
import pytest


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
    mock_cluster,
    mock_read_csv,
):
    # Arrange
    mock_get_default_az_cred.return_value = "test_cred"
    mock_df = MagicMock()
    mock_generate_qna.return_value = mock_df
    mock_cluster = MagicMock()
    mock_read_csv = MagicMock()
    mock_config.SAMPLE_DATA = False
    all_docs = [
        {
            "text1": "Pigeons, also known as rock doves, are a common sight in urban areas around the world. These birds are known for their distinctive cooing call and their ability to navigate long distances. Pigeons are also appreciated for their beauty, with their colorful feathers and iridescent sheen."
        },
        {
            "text2": "Pigeons have been domesticated for thousands of years and have been used for a variety of purposes, including delivering messages during wartime and racing competitions. They are also popular as pets and can be trained to perform tricks."
        },
        {
            "text3": "Despite their reputation as pests, pigeons play an important role in the ecosystem. They help to spread seeds and nutrients throughout their environment and are even considered a keystone species in some areas."
        },
    ]
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
    mock_cluster.assert_called_once_with(all_docs, "test_dir", mock_config)
    mock_read_csv.assert_not_called()


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
