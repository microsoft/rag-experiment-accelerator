import unittest
from unittest.mock import patch, MagicMock, call
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.index_config import IndexConfig


class TestIndex(unittest.TestCase):
    @patch("rag_experiment_accelerator.run.index.get_logger")
    @patch("rag_experiment_accelerator.run.index.Environment")
    @patch("rag_experiment_accelerator.run.index.Config")
    @patch("rag_experiment_accelerator.run.index.load_documents")
    @patch("rag_experiment_accelerator.run.index.upload_data")
    @patch("rag_experiment_accelerator.run.index.create_acs_index")
    @patch("rag_experiment_accelerator.run.index.Preprocess")
    @patch("rag_experiment_accelerator.embedding.embedding_model.EmbeddingModel")
    def test_run_with_config_values(
        self,
        mock_embedding_model,
        mock_preprocess,
        mock_create_acs_index,
        mock_upload_data,
        mock_load_documents,
        mock_config,
        mock_environment,
        mock_get_logger,
    ):
        # Create a list of mock EmbeddingModel instances
        embedding_models = [mock_embedding_model for _ in range(2)]

        # Set a side effect to assign a dimension value to each embedding model
        mock_embedding_model.side_effect = [
            MagicMock(dimension=100 * i) for i in range(1, 3)
        ]

        for embedding_model in embedding_models:
            index_config = IndexConfig(
                index_name_prefix="test_index_name",
                chunk_size="chunk_size1",
                overlap="overlap_size1",
                embedding_model=embedding_model,
                ef_construction="ef_construction1",
                ef_search="ef_search1",
            )

            self.run_test(
                index_config,
                mock_preprocess,
                mock_create_acs_index,
                mock_upload_data,
                mock_load_documents,
                mock_config,
                mock_environment,
                mock_get_logger,
            )

    def run_test(
        self,
        index_config,
        mock_preprocess,
        mock_create_acs_index,
        mock_upload_data,
        mock_load_documents,
        mock_config,
        mock_environment,
        mock_get_logger,
    ):
        # Arrange
        mock_config.DATA_FORMATS = "test_format"
        mock_config.artifacts_dir = "test_artifacts_dir"
        mock_config.data_dir = "data_dir"
        mock_config.AZURE_OAI_CHAT_DEPLOYMENT_NAME = "test_deployment_name"
        mock_config.CHUNKING_STRATEGY = "basic"
        mock_environment.azure_search_service_endpoint = "test_endpoint"
        mock_environment.azure_search_admin_key = "test_key"

        doc1 = MagicMock()
        doc1.page_content = "content1"
        doc2 = MagicMock()
        doc2.page_content = "content2"
        mock_load_documents.return_value = [doc1, doc2]

        # Mock the generate_embedding method for each embedding model
        index_config.embedding_model.generate_embedding = MagicMock(
            return_value="embedding_value"
        )

        chunks = [
            {
                "content": "content1",
                "content_vector": index_config.embedding_model.generate_embedding(),
            },
            {
                "content": "content2",
                "content_vector": index_config.embedding_model.generate_embedding(),
            },
        ]
        file_paths = ["file_path_one", "file_path_two"]

        # Act
        run(
            environment=mock_environment,
            config=mock_config,
            index_config=index_config,
            file_paths=file_paths,
        )

        # Assert
        # mock_config.assert_called_once()
        mock_load_documents.assert_called()
        expected_call = call(
            mock_environment,
            "basic",
            "test_format",
            file_paths,
            index_config.chunk_size,
            index_config.overlap,
        )
        mock_load_documents.assert_has_calls([expected_call])
        expected_first_call_args = [
            chunks,
            "test_endpoint",
            "test_index_name",
            "test_key",
            index_config.embedding_model,
            "test_deployment_name",
        ]
        _, kwargs = mock_upload_data.call_args
        # Assert that the call arguments of the first call are as expected
        self.assertEqual(kwargs.get("chunks"), expected_first_call_args[0])
        self.assertEqual(kwargs.get("service_endpoint"), expected_first_call_args[1])
        # self.assertEqual(kwargs.get('index_name'), expected_first_call_args[2])
        self.assertEqual(kwargs.get("search_key"), expected_first_call_args[3])
        self.assertEqual(kwargs.get("embedding_model"), expected_first_call_args[4])
        self.assertEqual(
            kwargs.get("azure_oai_deployment_name"), expected_first_call_args[5]
        )
        mock_create_acs_index.assert_called()
        # TODO
        # mock_preprocess.assert_called_once()
        mock_create_acs_index.assert_called()

    @patch("rag_experiment_accelerator.run.index.create_acs_index")
    @patch("rag_experiment_accelerator.run.index.load_documents")
    @patch("rag_experiment_accelerator.run.index.upload_data")
    @patch("rag_experiment_accelerator.run.index.Preprocess")
    @patch("rag_experiment_accelerator.run.index.Config")
    @patch("rag_experiment_accelerator.run.index.IndexConfig")
    @patch("rag_experiment_accelerator.run.index.Environment")
    def test_run_upload_data_exception(
        self,
        mock_environment,
        mock_index_config,
        mock_config,
        mock_preprocess,
        mock_upload_data,
        mock_load_documents,
        mock_create_acs_index,
    ):
        # Arrange
        mock_upload_data.side_effect = Exception("Test exception")

        # Act
        with self.assertRaises(Exception) as context:
            run(
                environment=mock_environment,
                config=mock_config,
                index_config=mock_index_config,
                file_paths=["file_path_one", "file_path_two"],
            )

        # Assert
        self.assertTrue("Test exception" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
