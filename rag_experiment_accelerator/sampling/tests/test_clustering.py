import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.sampling.clustering import cluster


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "text": ["This is document 1", "This is document 2", "This is document 3"],
            "processed_text": ["this document 1", "this document 2", "this document 3"],
        }
    )


@pytest.fixture
def mock_reducer():
    return MagicMock()


@pytest.fixture
def mock_df_concat():
    return pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "text": ["This is document 1", "This is document 2", "This is document 3"],
            "prediction": [0, 1, 0],
            "chunk": [0, 1, 0],
        }
    )


@pytest.fixture
def mock_data_dir(tmpdir):
    return tmpdir.mkdir("data")


def test_cluster(mock_logger, mock_df, mock_reducer, mock_df_concat, mock_data_dir):
    # Arrange
    all_chunks = [
        {"text": "This is document 1"},
        {"text": "This is document 2"},
        {"text": "This is document 3"},
    ]
    config = MagicMock()
    config.SAMPLE_OPTIMUM_K = 2
    config.SAMPLE_MIN_CLUSTER = 1
    config.SAMPLE_MAX_CLUSTER = 10
    config.SAMPLE_PERCENTAGE = 50

    with patch(
        "rag_experiment_accelerator.sampling.clustering.logger", mock_logger
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.chunk_dict_to_dataframe",
        return_value=mock_df,
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.vectorize_tfidf"
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.UMAP", return_value=mock_reducer
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.determine_optimum_k_elbow",
        return_value=2,
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.cluster_kmeans",
        return_value=(0, 0, "text", "processed_text", 0, [0, 1], [0.5, 0.6]),
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.pd.DataFrame",
        return_value=mock_df_concat,
    ):
        # Act
        result = cluster(all_chunks, mock_data_dir, config)

        # Assert
        assert len(result) == 2
        assert os.path.exists(
            f"{mock_data_dir}/sampling/all_cluster_predictions_cluster_number_2.csv"
        )
        assert os.path.exists(
            f"{mock_data_dir}/sampling/sampled_cluster_predictions_cluster_number_2.csv"
        )
        assert mock_logger.info.call_count == 4
        assert (
            mock_logger.info.call_args_list[0][0][0]
            == "Sampling - Original Document chunk length 3"
        )
        assert mock_logger.info.call_args_list[1][0][0] == "Run TF-IDF"
        assert mock_logger.info.call_args_list[2][0][0] == "Reducing Umap"
        assert (
            mock_logger.info.call_args_list[3][0][0]
            == "Sampled Document chunk length 2"
        )
