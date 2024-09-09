import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.checkpoint import init_checkpoint
from rag_experiment_accelerator.sampling.clustering import cluster, load_parser


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "text": [
                "Pigeons, also known as rock doves, are a common sight in urban areas around the world. These birds "
                "are known for their distinctive cooing call and their ability to navigate long distances. Pigeons "
                "are also appreciated for their beauty, with their colorful feathers and iridescent sheen.",
                "Pigeons have been domesticated for thousands of years and have been used for a variety of purposes, "
                "including delivering messages during wartime and racing competitions. They are also popular as pets "
                "and can be trained to perform tricks.",
                "Despite their reputation as pests, pigeons play an important role in the ecosystem. They help to "
                "spread seeds and nutrients throughout their environment and are even considered a keystone species "
                "in some areas.",
            ],
            "processed_text": [
                "Pigeons, also known as rock doves, are a common sight in urban areas around the world. These birds "
                "are known for their distinctive cooing call and their ability to navigate long distances. Pigeons "
                "are also appreciated for their beauty, with their colorful feathers and iridescent sheen.",
                "Pigeons have been domesticated for thousands of years and have been used for a variety of purposes, "
                "including delivering messages during wartime and racing competitions. They are also popular as pets "
                "and can be trained to perform tricks.",
                "Despite their reputation as pests, pigeons play an important role in the ecosystem. They help to "
                "spread seeds and nutrients throughout their environment and are even considered a keystone species "
                "in some areas.",
            ],
        }
    )


@pytest.fixture
def mock_reducer():
    return MagicMock()


@pytest.fixture
def mock_df_concat():
    return pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4],
            "y": [0, 1, 2, 3, 4],
            "text": [
                "Pigeons, also known as rock doves, are a common sight in urban areas around the world. These birds are known for their distinctive cooing call and their ability to navigate long distances. Pigeons are also appreciated for their beauty, with their colorful feathers and iridescent sheen.",
                "Pigeons have been domesticated for thousands of years and have been used for a variety of purposes, including delivering messages during wartime and racing competitions. They are also popular as pets and can be trained to perform tricks.",
                "Despite their reputation as pests, pigeons play an important role in the ecosystem. They help to spread seeds and nutrients throughout their environment and are even considered a keystone species in some areas.",
                "Overall, pigeons are fascinating and complex creatures that have captured the attention of people for centuries. Whether you love them or hate them, there is no denying the impact that pigeons have had on human society and the natural world.",
                "However, pigeons can also be carriers of diseases and can cause damage to buildings and other structures. It is important to take proper precautions when dealing with pigeons, such as wearing gloves and avoiding direct contact with their droppings.",
            ],
            "prediction": [0, 1, 0, 1, 0],
            "chunk": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def mock_data_dir(tmpdir):
    return tmpdir.mkdir("data")


def test_cluster(mock_logger, mock_df, mock_reducer, mock_df_concat, mock_data_dir):
    # Arrange
    all_chunks = [
        {
            "content": "Pigeons, also known as rock doves, are a common sight in urban areas around the world. These birds are known for their distinctive cooing call and their ability to navigate long distances. Pigeons are also appreciated for their beauty, with their colorful feathers and iridescent sheen.",
            "metadata": {"source": mock_data_dir + "/sampling/tests/data/test1.txt"},
        },
        {
            "content": "Pigeons have been domesticated for thousands of years and have been used for a variety of purposes, including delivering messages during wartime and racing competitions. They are also popular as pets and can be trained to perform tricks.",
            "metadata": {"source": mock_data_dir + "/sampling/tests/data/test2.txt"},
        },
        {
            "content": "Despite their reputation as pests, pigeons play an important role in the ecosystem. They help to spread seeds and nutrients throughout their environment and are even considered a keystone species in some areas.",
            "metadata": {"source": mock_data_dir + "/sampling/tests/data/test3.txt"},
        },
    ]

    config = MagicMock()
    config.use_checkpoints = False
    config.index.sampling.optimum_k = 2
    config.index.sampling.min_cluster = 1
    config.index.sampling.max_cluster = 10
    config.index.sampling.percentage = 100
    config.path.sampling_output_dir = os.path.join(mock_data_dir, "sampling")
    os.makedirs(config.path.sampling_output_dir)

    sampled_input_data_csv_path = f"{config.path.sampling_output_dir}/sampled_cluster_predictions_cluster_number_{config.index.sampling.optimum_k}.csv"
    config.path.sampled_cluster_predictions_path.return_value = (
        sampled_input_data_csv_path
    )

    init_checkpoint(config)

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
        return_value=(
            0,
            0,
            "text",
            "processed_text",
            0,
            [0, 1],
            [0.5, 0.6],
            [
                mock_data_dir + "/sampling/tests/data/test1.txt",
                mock_data_dir + "/sampling/tests/data/test2.txt",
                mock_data_dir + "/sampling/tests/data/test3.txt",
            ],
        ),
    ), patch(
        "rag_experiment_accelerator.sampling.clustering.pd.DataFrame",
        return_value=mock_df_concat,
    ):
        # Act
        parser = load_parser()
        result = cluster("", all_chunks, config, parser)
        assert len(result) == 0
        # Assert
        assert os.path.exists(
            os.path.join(
                config.path.sampling_output_dir,
                "all_cluster_predictions_cluster_number_2.csv",
            )
        )
        assert os.path.exists(
            os.path.join(
                config.path.sampling_output_dir,
                "sampled_cluster_predictions_cluster_number_2.csv",
            )
        )
        assert (
            mock_logger.info.call_args_list[0][0][0]
            == "Sampling - Original Document chunk length 3"
        )
        assert mock_logger.info.call_args_list[1][0][0] == "Run TF-IDF"
        assert mock_logger.info.call_args_list[2][0][0] == "Reducing Umap"
        assert (
            mock_logger.info.call_args_list[3][0][0]
            == "Sampled Document chunk length 0"
        )
