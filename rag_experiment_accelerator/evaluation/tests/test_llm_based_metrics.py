from unittest.mock import patch

from rag_experiment_accelerator.evaluation.ragas_metrics import RagasEvals


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
@patch("rag_experiment_accelerator.evaluation.llm_based_metrics.SentenceTransformer")
def test_ragas_answer_relevance(mock_st, mock_generate_response):
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
    eval = RagasEvals(mock_generate_response)
    score = eval.ragas_answer_relevance(mock_generate_response, question, answer)
    assert round(score) == 100


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_ragas_context_precision(mock_generate_response):
    question = "What is the name of the largest bone in the human body?"
    retrieved_contexts = ["Retrieved context 1", "Retrieved context 2"]
    mock_generate_response.generate_response.side_effect = ["Yes", "No", "Yes", "No"]

    eval = RagasEvals(mock_generate_response)
    score = eval.ragas_context_precision(mock_generate_response, question, retrieved_contexts)

    expected_relevancy_scores = [1, 0, 1, 0]
    expected_precision = (
        sum(expected_relevancy_scores) / len(expected_relevancy_scores)
    ) * 100  # 50.0

    assert score == expected_precision


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_ragas_context_recall(mock_generate_response):
    mock_generate_response.generate_response.return_value = (
        '"Attributed": "1"   "Attributed": "1"   "Attributed": "1"   "Attributed": "0"'
    )
    question = "What is the name of the largest bone in the human body?"
    context = 'According to the Cleveland Clinic, "The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average adult male femur is 48 cm (18.9 in) in length and 2.34 cm (0.92 in) in diameter. The average weight among adult males in the United States is 196 lbs (872 N). Therefore, the adult male femur can support roughly 6,000 lbs of compressive force."'
    answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."

    eval = RagasEvals(mock_generate_response)
    score = eval.ragas_context_recall(mock_generate_response, question, answer, context)
    assert score == 75
