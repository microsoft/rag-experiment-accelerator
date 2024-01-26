import unittest
import os
from unittest.mock import MagicMock, patch
from azure.search.documents import SearchClient
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.run.querying import query_acs, rerank_documents, run


class TestQuerying(unittest.TestCase):
    @patch("rag_experiment_accelerator.run.querying.search_mapping")
    def test_query_acs(self, mock_search_mapping):
        mock_search_client = MagicMock(spec=SearchClient)
        mock_embedding_model = MagicMock(spec=EmbeddingModel)
        user_prompt = "test prompt"
        s_v = "search_for_match_semantic"
        retrieve_num_of_documents = "10"

        query_acs(
            mock_search_client,
            mock_embedding_model,
            user_prompt,
            s_v,
            retrieve_num_of_documents,
        )

        mock_search_mapping[s_v].assert_called_once_with(
            client=mock_search_client,
            embedding_model=mock_embedding_model,
            query=user_prompt,
            retrieve_num_of_documents=retrieve_num_of_documents,
        )

    @patch("rag_experiment_accelerator.run.querying.llm_rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.cross_encoder_rerank_documents")
    def test_rerank_documents(
        self, mock_cross_encoder_rerank_documents, mock_llm_rerank_documents
    ):
        docs = ["doc1", "doc2"]
        user_prompt = "test prompt"
        output_prompt = "output prompt"
        config = Config()
        config.RERANK_TYPE = "llm"

        rerank_documents(docs, user_prompt, output_prompt, config)

        mock_llm_rerank_documents.assert_called_once()

    # create a test for run
    @patch("rag_experiment_accelerator.run.querying.Config")
    @patch("rag_experiment_accelerator.run.querying.get_default_az_cred")
    @patch("rag_experiment_accelerator.run.querying.SpacyEvaluator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutputHandler")
    @patch("rag_experiment_accelerator.run.querying.create_client")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutput")
    @patch("rag_experiment_accelerator.run.querying.create_data_asset")
    @patch("rag_experiment_accelerator.run.querying.do_we_need_multiple_questions")
    def test_run_output(
        self,
        mock_do_we_need_multiple_questions,
        mock_create_data_asset,
        mock_query_output,
        mock_response_generator,
        mock_create_client,
        mock_query_output_handler,
        mock_spacy_evaluator,
        mock_get_default_az_cred,
        mock_config,
    ):
        # Arrange
        mock_query_output_handler.return_value.load.return_value = [mock_query_output]
        mock_query_output_handler.return_value.save.side_effect = None
        mock_config.return_value.CHUNK_SIZES = [1]
        mock_config.return_value.OVERLAP_SIZES = [1]
        mock_config.return_value.RERANK_TYPE = "llm"
        mock_config.return_value.RETRIEVE_NUM_OF_DOCUMENTS = 1
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = test_dir + "/data/test_data.jsonl"
        mock_config.return_value.EVAL_DATA_JSONL_FILE_PATH = data_file_path
        mock_embedding_model = MagicMock(spec=EmbeddingModel)
        mock_embedding_model.name = "test-embedding-model"
        mock_config.return_value.embedding_models = [mock_embedding_model]
        mock_config.return_value.EF_CONSTRUCTIONS = [400]
        mock_config.return_value.EF_SEARCHES = [400]
        mock_config.return_value.SEARCH_VARIANTS = ["search_for_match_semantic"]
        mock_config.return_value.NAME_PREFIX = "prefix"
        # Act
        run("config_dir")

        # Assert
        mock_query_output_handler.save.assert_called()
        mock_create_data_asset.assert_called()


if __name__ == "__main__":
    unittest.main()
