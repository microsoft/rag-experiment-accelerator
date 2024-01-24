import unittest
from unittest.mock import patch, MagicMock, call
from rag_experiment_accelerator.run.index import run


class TestIndex(unittest.TestCase):
    @patch('rag_experiment_accelerator.run.index.load_dotenv')
    @patch('rag_experiment_accelerator.run.index.get_logger')
    @patch('rag_experiment_accelerator.run.index.Config')
    @patch('rag_experiment_accelerator.run.index.load_documents')
    @patch('rag_experiment_accelerator.run.index.upload_data')
    @patch('rag_experiment_accelerator.run.index.create_acs_index')
    @patch('rag_experiment_accelerator.run.index.Preprocess')
    @patch('rag_experiment_accelerator.run.index.get_index_name')
    @patch('rag_experiment_accelerator.embedding.embedding_model.EmbeddingModel')
    def test_run_with_config_values(self, mock_embedding_model, mock_get_index_name, mock_Preprocess, mock_create_acs_index, mock_upload_data, mock_load_documents, mock_Config, mock_get_logger, mock_load_dotenv):
        # Create a list of mock EmbeddingModel instances
        embedding_models = [mock_embedding_model for _ in range(2)]

        # Set a side effect to assign a dimension value to each embedding model
        mock_embedding_model.side_effect = [
            MagicMock(dimension=100 * i) for i in range(1, 3)]

        self.run_test(['chunk_size1'], ['overlap_size1'], embedding_models, ['ef_construction1'], ['ef_search1'], mock_get_index_name,
                      mock_Preprocess, mock_create_acs_index, mock_upload_data, mock_load_documents, mock_Config, mock_get_logger, mock_load_dotenv)

    def run_test(self, chunk_sizes, overlap_sizes, embedding_models, ef_constructions, ef_searches, mock_get_index_name, mock_Preprocess, mock_create_acs_index, mock_upload_data, mock_load_documents, mock_Config, mock_get_logger, mock_load_dotenv):
        # Arrange
        mock_get_index_name.return_value = 'test_index_name'
        mock_Config.return_value.CHUNK_SIZES = chunk_sizes
        mock_Config.return_value.OVERLAP_SIZES = overlap_sizes
        mock_Config.return_value.embedding_models = embedding_models
        mock_Config.return_value.EF_CONSTRUCTIONS = ef_constructions
        mock_Config.return_value.EF_SEARCHES = ef_searches
        mock_Config.return_value.DATA_FORMATS = 'test_format'
        mock_Config.return_value.artifacts_dir = 'test_artifacts_dir'
        mock_Config.return_value.data_dir = 'data_dir'
        mock_Config.return_value.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT = 'test_endpoint'
        mock_Config.return_value.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY = 'test_key'
        mock_Config.return_value.AZURE_OAI_CHAT_DEPLOYMENT_NAME = 'test_deployment_name'
        doc1 = MagicMock()
        doc1.page_content = 'content1'
        doc2 = MagicMock()
        doc2.page_content = 'content2'
        mock_load_documents.return_value = [doc1, doc2]

        # Mock the generate_embedding method for each embedding model
        for model in embedding_models:
            model.generate_embedding = MagicMock(
                return_value='embedding_value')

        chunks = [
            {'content': 'content1',
                'content_vector': embedding_models[0].generate_embedding()},
            {'content': 'content2',
                'content_vector': embedding_models[1].generate_embedding()}
        ]

        # Act
        run('config_dir', 'data_dir')

        # Assert
        mock_Config.assert_called_once()
        mock_load_documents.assert_called()
        expected_calls = [call('test_format', 'data_dir', chunk_size, overlap_size)
                          for chunk_size, overlap_size in zip(chunk_sizes, overlap_sizes)]
        mock_load_documents.assert_has_calls(expected_calls, any_order=True)
        expected_first_call_args = [chunks, 'test_endpoint', 'test_index_name',
                                    'test_key', embedding_models[0], 'test_deployment_name']
        args, kwargs = mock_upload_data.call_args
        # Assert that the call arguments of the first call are as expected
        self.assertEqual(kwargs.get('chunks'), expected_first_call_args[0])
        self.assertEqual(kwargs.get('service_endpoint'),
                         expected_first_call_args[1])
        self.assertEqual(kwargs.get('index_name'), expected_first_call_args[2])
        self.assertEqual(kwargs.get('search_key'), expected_first_call_args[3])
        self.assertEqual(kwargs.get('embedding_model'),
                         expected_first_call_args[4])
        self.assertEqual(kwargs.get('azure_oai_deployment_name'),
                         expected_first_call_args[5])
        mock_create_acs_index.assert_called()
        mock_Preprocess.assert_called_once()
        mock_get_index_name.assert_called()
        mock_create_acs_index.assert_called()

        @patch('rag_experiment_accelerator.run.index.os.makedirs')
        @patch('rag_experiment_accelerator.run.index.create_acs_index')
        @patch('rag_experiment_accelerator.run.index.get_index_name')
        @patch('rag_experiment_accelerator.run.index.load_documents')
        @patch('rag_experiment_accelerator.run.index.upload_data')
        @patch('rag_experiment_accelerator.run.index.Preprocess')
        @patch('rag_experiment_accelerator.run.index.Config')
        def test_run_makedirs_exception(self, mock_config, mock_preprocess, mock_upload_data, mock_load_documents, mock_get_index_name, mock_create_acs_index, mock_makedirs):
            # Arrange
            mock_makedirs.side_effect = Exception('Test exception')

            # Act
            from rag_experiment_accelerator.run.index import run
            with self.assertRaises(Exception) as context:
                run('config_dir', 'data_dir')

            # Assert
            self.assertTrue('Test exception' in str(context.exception))

        @patch('rag_experiment_accelerator.run.index.create_acs_index')
        @patch('rag_experiment_accelerator.run.index.get_index_name')
        @patch('rag_experiment_accelerator.run.index.load_documents')
        @patch('rag_experiment_accelerator.run.index.upload_data')
        @patch('rag_experiment_accelerator.run.index.Preprocess')
        @patch('rag_experiment_accelerator.run.index.Config')
        def test_run_upload_data_exception(self, mock_config, mock_preprocess, mock_upload_data, mock_load_documents, mock_get_index_name, mock_create_acs_index):
            # Arrange
            mock_upload_data.side_effect = Exception('Test exception')

            # Act
            with self.assertRaises(Exception) as context:
                run('config_dir', 'data_dir')

            # Assert
            self.assertTrue('Test exception' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
