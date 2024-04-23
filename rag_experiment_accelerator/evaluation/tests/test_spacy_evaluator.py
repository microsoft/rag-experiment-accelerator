from unittest.mock import MagicMock, call, patch
from rag_experiment_accelerator.evaluation.spacy_evaluator import (
    SpacyEvaluator,
)


@patch("rag_experiment_accelerator.evaluation.spacy_evaluator.load")
def test_evaluator_init(mock_nlp):
    similarity_threshold = 0.4
    evaluator = SpacyEvaluator(similarity_threshold=similarity_threshold)
    assert similarity_threshold == evaluator.similarity_threshold


@patch("rag_experiment_accelerator.evaluation.spacy_evaluator.load")
def test_similarity_returns_similar(mock_nlp):
    mock_doc_1 = MagicMock()
    mock_doc_1.similarity.return_value = 1
    mock_doc_2 = MagicMock()
    mock_nlp().side_effect = [mock_doc_1, mock_doc_2]

    evaluator = SpacyEvaluator()
    actual = evaluator.similarity("test word", "test word")

    mock_doc_1.similarity.assert_called_once_with(mock_doc_2)
    assert actual == 1


@patch(
    "rag_experiment_accelerator.evaluation.spacy_evaluator.SpacyEvaluator.similarity"
)
@patch("rag_experiment_accelerator.evaluation.spacy_evaluator.load")
def test_is_relevant_returns_valid(mock_nlp, mock_similarity):
    mock_similarity.side_effect = [1, 0.05]

    evaluator = SpacyEvaluator()
    actual_true = evaluator.is_relevant("test phrase", "test phrase")
    actual_false = evaluator.is_relevant("phrase", "different")

    mock_similarity.assert_has_calls(
        [call("test phrase", "test phrase"), call("phrase", "different")]
    )
    assert actual_true is True
    assert actual_false is False
