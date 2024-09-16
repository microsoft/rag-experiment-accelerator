from unittest.mock import patch

from rag_experiment_accelerator.evaluation.search_eval import (
    evaluate_search_result,
)

evaluation_content = "my content to evaluate"
search_response = [
    {
        "@search.score": 0.03755760192871094,
        "content": "this is the first chunk",
    },
    {
        "@search.score": 0.029906954616308212,
        "content": "this is the second chunk",
    },
    {
        "@search.score": 0.028612013906240463,
        "content": "this is the third chunk",
    },
]


def test_evaluate_search_result_calulates_precision_score():
    with patch(
        "rag_experiment_accelerator.evaluation.spacy_evaluator.SpacyEvaluator"
    ) as evaluator:
        evaluator.is_relevant.side_effect = [True, False, True]

        _, evaluation = evaluate_search_result(
            search_response, evaluation_content, evaluator
        )

        expected_precision = [1.0, 0.5, 0.67]
        for i, precision in enumerate(evaluation.get("precision_scores")):
            assert precision == expected_precision[i]


def test_evaluate_search_result_calulates_recall_score():
    with patch(
        "rag_experiment_accelerator.evaluation.spacy_evaluator.SpacyEvaluator"
    ) as evaluator:
        evaluator.is_relevant.side_effect = [True, False, True]

        _, evaluation = evaluate_search_result(
            search_response, evaluation_content, evaluator
        )

        expected_recall = [0.5, 0.5, 1.0]
        for i, recall in enumerate(evaluation.get("recall_scores")):
            assert recall == expected_recall[i]


def test_evaluate_search_result_returns_all_search_content():
    with patch(
        "rag_experiment_accelerator.evaluation.spacy_evaluator.SpacyEvaluator"
    ) as evaluator:
        evaluator.is_relevant.side_effect = [True, False, True]

        content, _ = evaluate_search_result(
            search_response, evaluation_content, evaluator
        )

        for i, doc in enumerate(search_response):
            assert doc["content"] == content[i]
