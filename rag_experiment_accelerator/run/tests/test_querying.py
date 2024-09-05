import unittest
import os
from unittest.mock import MagicMock, patch
from azure.search.documents import SearchClient
from rag_experiment_accelerator.checkpoint import init_checkpoint
from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.openai_config import OpenAIConfig
from rag_experiment_accelerator.config.path_config import PathConfig
from rag_experiment_accelerator.config.query_expansion import QueryExpansionConfig
from rag_experiment_accelerator.config.rerank_config import RerankConfig
from rag_experiment_accelerator.config.search_config import SearchConfig
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.run.querying import (
    QueryAndEvalACSResult,
    query_acs,
    query_and_eval_single_line,
    rerank_documents,
    query_and_eval_acs,
    query_and_eval_acs_multi,
)
from rag_experiment_accelerator.llm.prompt import Prompt, main_instruction


class TestQuerying(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock(spec=Config)

        self.mock_config.use_checkpoints = False

        self.mock_config.index = MagicMock(spec=IndexConfig)
        self.mock_config.index.index_name_prefix = "prefix"
        self.mock_config.index.ef_construction = [400]
        self.mock_config.index.ef_search = [400]
        self.mock_config.index.chunking = MagicMock(spec=ChunkingConfig)
        self.mock_config.index.chunking.chunk_size = [1]
        self.mock_config.index.chunking.overlap_size = [1]
        self.mock_config.index.embedding_model = MagicMock(spec=EmbeddingModel)
        self.mock_config.index.embedding_model.model_name = "test-embedding-model"

        self.mock_config.query_expansion = MagicMock(spec=QueryExpansionConfig)
        self.mock_config.query_expansion.query_expansion = False
        self.mock_config.query_expansion.hyde = "disabled"
        self.mock_config.query_expansion.min_query_expansion_related_question_similarity_score = (
            90
        )
        self.mock_config.query_expansion.expand_to_multiple_questions = True

        self.mock_config.openai = MagicMock(spec=OpenAIConfig)
        self.mock_config.openai.azure_oai_chat_deployment_name = "test-deployment"

        self.mock_config.rerank = MagicMock(spec=RerankConfig)
        self.mock_config.rerank.enabled = True
        self.mock_config.rerank.type = "llm"
        self.mock_config.rerank.llm_rerank_threshold = 3

        self.mock_config.search = MagicMock(spec=SearchConfig)
        self.mock_config.search.retrieve_num_of_documents = 10
        self.mock_config.search.search_type = ["search_for_match_semantic"]

        self.mock_config.path = MagicMock(spec=PathConfig)
        self.mock_config.path.eval_data_file = "test_data.jsonl"

        self.mock_environment = MagicMock(spec=Environment)
        self.mock_search_client = MagicMock(spec=SearchClient)
        self.mock_embedding_model = MagicMock(spec=EmbeddingModel)

        self.prompt = MagicMock(spec=Prompt)
        self.prompt.tags = {}
        self.prompt.system_message = "system message"
        self.prompt.user_template = "user template"

    @patch("rag_experiment_accelerator.run.querying.search_mapping")
    def test_query_acs(self, mock_search_mapping):
        user_prompt = "test prompt"
        s_v = "search_for_match_semantic"
        retrieve_num_of_documents = "10"

        query_acs(
            self.mock_search_client,
            self.mock_embedding_model,
            user_prompt,
            s_v,
            retrieve_num_of_documents,
        )

        mock_search_mapping[s_v].assert_called_once_with(
            client=self.mock_search_client,
            embedding_model=self.mock_embedding_model,
            query=user_prompt,
            retrieve_num_of_documents=retrieve_num_of_documents,
        )

    @patch("rag_experiment_accelerator.run.querying.llm_rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.cross_encoder_rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    def test_rerank_documents(
        self,
        mock_response_generator,
        mock_cross_encoder_rerank_documents,
        mock_llm_rerank_documents,
    ):
        docs = ["doc1", "doc2"]
        user_prompt = "test prompt"
        output_prompt = "output prompt"

        rerank_documents(
            docs, user_prompt, output_prompt, self.mock_config, mock_response_generator
        )

        mock_llm_rerank_documents.assert_called_once()
        mock_cross_encoder_rerank_documents.assert_not_called()

    @patch("rag_experiment_accelerator.run.querying.query_acs")
    @patch("rag_experiment_accelerator.run.querying.evaluate_search_result")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    def test_query_and_eval_acs(
        self, mock_response_generator, mock_evaluate_search_result, mock_query_acs
    ):
        # Arrange
        query = "test query"
        search_type = "test search type"
        evaluation_content = "test evaluation content"
        retrieve_num_of_documents = 10
        mock_evaluator = MagicMock()
        mock_search_result = [
            {"content": "text1", "@search.score": 10},
            {"content": "text2", "@search.score": 9},
        ]
        mock_docs = ["doc1", "doc2"]
        mock_evaluation = {"score": 0.8}

        mock_query_acs.return_value = mock_search_result
        mock_evaluate_search_result.return_value = (mock_docs, mock_evaluation)

        # Act
        result = query_and_eval_acs(
            self.mock_search_client,
            self.mock_embedding_model,
            query,
            search_type,
            evaluation_content,
            retrieve_num_of_documents,
            mock_evaluator,
            self.mock_config,
            mock_response_generator(),
        )

        # Assert
        mock_query_acs.assert_called_once_with(
            search_client=self.mock_search_client,
            embedding_model=self.mock_embedding_model,
            user_prompt=query,
            s_v=search_type,
            retrieve_num_of_documents=retrieve_num_of_documents,
        )
        mock_evaluate_search_result.assert_called_once_with(
            mock_search_result, evaluation_content, mock_evaluator
        )
        self.assertEqual(result.documents, mock_docs)
        self.assertEqual(result.evaluations, mock_evaluation)

    @patch("rag_experiment_accelerator.run.querying.query_and_eval_acs")
    @patch("rag_experiment_accelerator.run.querying.rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    def test_query_and_eval_acs_multi_rerank(
        self,
        mock_response_generator,
        mock_rerank_documents,
        mock_query_and_eval_acs,
    ):
        # Arrange
        questions = ["question1", "question2"]
        original_prompt = "original prompt"
        output_prompt = "output prompt"
        search_type = "search type"
        evaluation_content = "evaluation content"
        evaluator = MagicMock()
        mock_docs = ["doc1", "doc2"]
        mock_evaluation = {"score": 0.8}

        mock_query_and_eval_acs.side_effect = [
            QueryAndEvalACSResult(mock_docs, mock_evaluation),
            QueryAndEvalACSResult(mock_docs, mock_evaluation),
        ]
        mock_rerank_documents.return_value = prompt_instruction_context = [
            "context1",
            "context2",
        ]
        mock_response_generator.return_value.generate_response.return_value = (
            "openai response"
        )

        # Act
        result = query_and_eval_acs_multi(
            self.mock_search_client,
            self.mock_embedding_model,
            questions,
            original_prompt,
            output_prompt,
            search_type,
            evaluation_content,
            self.mock_config,
            evaluator,
            mock_response_generator(),
        )

        # Assert
        mock_query_and_eval_acs.assert_called_with(
            search_client=self.mock_search_client,
            embedding_model=self.mock_embedding_model,
            query=questions[1] or questions[0],
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=self.mock_config.search.retrieve_num_of_documents,
            evaluator=evaluator,
            config=self.mock_config,
            response_generator=mock_response_generator(),
        )
        # mock_rerank_documents.assert_not_called()
        mock_rerank_documents.assert_called_with(
            mock_docs,
            questions[1] or questions[0],
            output_prompt,
            self.mock_config,
            mock_response_generator(),
        )
        mock_response_generator.return_value.generate_response.assert_called_with(
            main_instruction,
            context="\n".join(prompt_instruction_context),
            question=original_prompt,
        )
        self.assertEqual(result.documents, ["openai response", "openai response"])
        self.assertEqual(result.evaluations, [mock_evaluation, mock_evaluation])

    @patch("rag_experiment_accelerator.run.querying.query_and_eval_acs")
    @patch("rag_experiment_accelerator.run.querying.rerank_documents")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    def test_query_and_eval_acs_multi_no_rerank(
        self,
        mock_response_generator,
        mock_rerank_documents,
        mock_query_and_eval_acs,
    ):
        # Arrange
        questions = ["question1", "question2"]
        original_prompt = "original prompt"
        output_prompt = "output prompt"
        search_type = "search type"
        evaluation_content = "evaluation content"
        self.mock_config.rerank = MagicMock(spec=RerankConfig)
        self.mock_config.rerank.enabled = False
        evaluator = MagicMock()
        mock_docs = ["doc1", "doc2"]
        mock_evaluation = {"score": 0.8}

        mock_query_and_eval_acs.side_effect = [
            QueryAndEvalACSResult(mock_docs, mock_evaluation),
            QueryAndEvalACSResult(mock_docs, mock_evaluation),
        ]

        mock_response_generator.return_value.generate_response.return_value = (
            "openai response"
        )

        # Act
        result = query_and_eval_acs_multi(
            self.mock_search_client,
            self.mock_embedding_model,
            questions,
            original_prompt,
            output_prompt,
            search_type,
            evaluation_content,
            self.mock_config,
            evaluator,
            response_generator=mock_response_generator(),
        )

        # Assert
        mock_query_and_eval_acs.assert_called_with(
            search_client=self.mock_search_client,
            embedding_model=self.mock_embedding_model,
            query=questions[1] or questions[0],
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=self.mock_config.search.retrieve_num_of_documents,
            evaluator=evaluator,
            config=self.mock_config,
            response_generator=mock_response_generator(),
        )
        mock_rerank_documents.assert_not_called()
        mock_response_generator.return_value.generate_response.assert_called_with(
            main_instruction,
            context="\n".join(mock_docs),
            question=original_prompt,
        )
        self.assertEqual(result.documents, ["openai response", "openai response"])
        self.assertEqual(result.evaluations, [mock_evaluation, mock_evaluation])

    @patch("rag_experiment_accelerator.run.querying.Environment")
    @patch("rag_experiment_accelerator.run.querying.SpacyEvaluator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutputHandler")
    @patch("rag_experiment_accelerator.run.querying.ResponseGenerator")
    @patch("rag_experiment_accelerator.run.querying.QueryOutput")
    @patch("rag_experiment_accelerator.run.querying.do_we_need_multiple_questions")
    @patch("rag_experiment_accelerator.run.querying.query_and_eval_acs")
    def test_run_no_multi_no_rerank(
        self,
        mock_query_and_eval_acs,
        mock_do_we_need_multiple_questions,
        mock_query_output,
        mock_response_generator,
        mock_query_output_handler,
        mock_spacy_evaluator,
        mock_environment,
    ):
        # Arrange
        mock_query_output_handler.return_value.load.return_value = [mock_query_output]
        mock_query_output_handler.return_value.save.side_effect = None
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_path = test_dir + "/data/test_data.jsonl"
        self.mock_config.path.eval_data_file = data_file_path
        self.mock_config.rerank = MagicMock(spec=RerankConfig)
        self.mock_config.rerank.enabled = False
        mock_do_we_need_multiple_questions.return_value = False
        mock_query_and_eval_acs.return_value = MagicMock()
        mock_search_client = MagicMock(SearchClient)

        init_checkpoint(self.mock_config)
        # Act
        with open(data_file_path, "r") as file:
            line = file.readline()
        query_and_eval_single_line(
            line,
            1,
            mock_query_output_handler,
            mock_environment,
            self.mock_config,
            self.mock_config.index,
            mock_response_generator,
            mock_search_client,
            mock_spacy_evaluator,
            1,
        )

        # Assert
        mock_query_and_eval_acs.assert_called()
        mock_query_output_handler.save.assert_called()


if __name__ == "__main__":
    unittest.main()
