from unittest.mock import patch

from rag_experiment_accelerator.evaluation.llm_based_metrics import (
    llm_answer_relevance,
    llm_context_precision,
    llm_context_recall,
)


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
@patch("rag_experiment_accelerator.evaluation.llm_based_metrics.SentenceTransformer")
def test_llm_answer_relevance(mock_st, mock_generate_response):
    mock_generate_response.return_value.generate_response.return_value = (
        "What is the name of the largest bone in the human body?"
    )
    mock_st().encode.side_effect = [[[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]]]

    question = "What is the name of the largest bone in the human body?"
    answer = (
        (
            "The largest bone in the human body is the femur, also known as"
            " the thigh bone. It is about 19.4 inches (49.5 cm) long on"
            " average and can support up to 30 times the weight of a person’s"
            " body."
        ),
    )
    score = llm_answer_relevance(mock_generate_response, question, answer)
    assert round(score) == 100


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_llm_context_precision(mock_generate_response):
    question = "What is the name of the largest bone in the human body?"
    retrieved_contexts = ["Retrieved context 1", "Retrieved context 2"]
    mock_generate_response.generate_response.side_effect = ["Yes", "No", "Yes", "No"]

    score = llm_context_precision(mock_generate_response, question, retrieved_contexts)

    expected_relevancy_scores = [1, 0, 1, 0]
    expected_precision = (
        sum(expected_relevancy_scores) / len(expected_relevancy_scores)
    ) * 100  # 50.0

    assert score == expected_precision


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_llm_context_recall(mock_generate_response):
    mock_generate_response.generate_response.return_value = [
        {
            "statement_1": "Test statement 1",
            "reason": "The statement is in the context",
            "attributed": "1",
        },
        {
            "statement_2": "Test statement 2",
            "reason": "The statement is in the context",
            "attributed": "1",
        },
        {
            "statement_3": "Test statement 3",
            "reason": "The statement is in the context",
            "attributed": "0",
        },
        {
            "statement_4": "Test statement 4",
            "reason": "The statement is in the context",
            "attributed": "1",
        },
    ]
    question = "What is the name of the largest bone in the human body?"
    context = 'According to the Cleveland Clinic, "The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average adult male femur is 48 cm (18.9 in) in length and 2.34 cm (0.92 in) in diameter. The average weight among adult males in the United States is 196 lbs (872 N). Therefore, the adult male femur can support roughly 6,000 lbs of compressive force."'
    answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."

    score = llm_context_recall(mock_generate_response, question, answer, context)
    assert score == 75
