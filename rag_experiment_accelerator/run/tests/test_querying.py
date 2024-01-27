import unittest
import os
from unittest.mock import MagicMock, patch
from azure.search.documents import SearchClient
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.run.querying import (
    query_acs,
    rerank_documents,
    query_and_eval_acs,
    run,
    query_and_eval_acs_multi,
)


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
        config = MagicMock()
        config.RERANK_TYPE = "llm"

        rerank_documents(docs, user_prompt, output_prompt, config)

        mock_llm_rerank_documents.assert_called_once()
        mock_cross_encoder_rerank_documents.assert_not_called()

    @patch("rag_experiment_accelerator.run.querying.query_acs")
    @patch("rag_experiment_accelerator.run.querying.evaluate_search_result")
    def test_query_and_eval_acs(self, mock_evaluate_search_result, mock_query_acs):
        # Arrange
        mock_search_client = MagicMock()
        mock_embedding_model = MagicMock()
        query = "test query"
        search_type = "test search type"
        evaluation_content = "test evaluation content"
        retrieve_num_of_documents = 10
        mock_evaluator = MagicMock()
        mock_search_result = MagicMock()
        mock_docs = ["doc1", "doc2"]
        mock_evaluation = {"score": 0.8}

        mock_query_acs.return_value = mock_search_result
        mock_evaluate_search_result.return_value = (mock_docs, mock_evaluation)

        # Act
        result_docs, result_evaluation = query_and_eval_acs(
            mock_search_client,
            mock_embedding_model,
            query,
            search_type,
            evaluation_content,
            retrieve_num_of_documents,
            mock_evaluator,
        )

        # Assert
        mock_query_acs.assert_called_once_with(
            search_client=mock_search_client,
            embedding_model=mock_embedding_model,
            user_prompt=query,
            s_v=search_type,
            retrieve_num_of_documents=retrieve_num_of_documents,
        )
        mock_evaluate_search_result.assert_called_once_with(
            mock_search_result, evaluation_content, mock_evaluator
        )
        self.assertEqual(result_docs, mock_docs)
        self.assertEqual(result_evaluation, mock_evaluation)

    @patch("rag_experiment_accelerator.run.querying.query_and_eval_acs")
    @patch("rag_experiment_accelerator.run.querying.rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    def test_query_and_eval_acs_multi(
        self,
        mock_response_generator,
        mock_rerank_documents,
        mock_query_and_eval_acs,
    ):
        # Arrange
        mock_search_client = MagicMock(spec=SearchClient)
        mock_embedding_model = MagicMock(spec=EmbeddingModel)
        questions = ["question1", "question2"]
        original_prompt = "original prompt"
        output_prompt = "output prompt"
        search_type = "search type"
        evaluation_content = "evaluation content"
        config = MagicMock(spec=Config)
        config.RETRIEVE_NUM_OF_DOCUMENTS = 10
        config.RERANK = False
        config.AZURE_OAI_CHAT_DEPLOYMENT_NAME = "test-deployment"
        evaluator = MagicMock()
        main_prompt_instruction = "main prompt instruction"
        mock_docs = ["doc1", "doc2"]
        mock_evaluation = {"score": 0.8}

        mock_query_and_eval_acs.side_effect = [
            (mock_docs, mock_evaluation),
            (mock_docs, mock_evaluation),
        ]
        mock_rerank_documents.return_value = prompt_instruction_context = [
            "context1",
            "context2",
        ]
        mock_response_generator.return_value.generate_response.return_value = (
            "openai response"
        )

        # Act
        result_context, result_evals = query_and_eval_acs_multi(
            mock_search_client,
            mock_embedding_model,
            questions,
            original_prompt,
            output_prompt,
            search_type,
            evaluation_content,
            config,
            evaluator,
            main_prompt_instruction,
        )

        # Assert
        mock_query_and_eval_acs.assert_called_with(
            search_client=mock_search_client,
            embedding_model=mock_embedding_model,
            query=questions[0],
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
            evaluator=evaluator,
        )
        mock_rerank_documents.assert_called_with(
            docs=mock_docs,
            user_prompt=questions[0],
            output_prompt=output_prompt,
            config=config,
        )
        mock_response_generator.return_value.generate_response.assert_called_with(
            main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context),
            original_prompt,
        )
        self.assertEqual(result_context, ["openai response", "openai response"])
        self.assertEqual(result_evals, [mock_evaluation, mock_evaluation])

    @patch("rag_experiment_accelerator.run.querying.Config")
    @patch("rag_experiment_accelerator.run.querying.get_default_az_cred")
    @patch("rag_experiment_accelerator.run.querying.SpacyEvaluator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutputHandler")
    @patch("rag_experiment_accelerator.run.querying.create_client")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutput")
    @patch("rag_experiment_accelerator.run.querying.create_data_asset")
    @patch("rag_experiment_accelerator.run.querying.do_we_need_multiple_questions")
    @patch("rag_experiment_accelerator.run.querying.query_and_eval_acs")
    def test_run_no_multi_no_rerank(
        self,
        mock_query_and_eval_acs,
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
        mock_config.return_value.RERANK = False
        mock_do_we_need_multiple_questions.return_value = False
        mock_query_and_eval_acs.return_value = [MagicMock(), MagicMock()]
        # Act
        run("test_config_dir")

        # Assert
        mock_query_and_eval_acs.assert_called()
        mock_response_generator.return_value.generate_response.assert_called()
        mock_query_output_handler.return_value.save.assert_called()
        mock_create_data_asset.assert_called()


if __name__ == "__main__":
    unittest.main()
