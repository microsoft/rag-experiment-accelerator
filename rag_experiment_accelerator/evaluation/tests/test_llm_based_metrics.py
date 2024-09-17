from unittest.mock import patch

from rag_experiment_accelerator.evaluation.ragas_metrics import RagasEvals
from rag_experiment_accelerator.evaluation.promptflow_quality_metrics import PromptFlowEvals


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
@patch("rag_experiment_accelerator.evaluation.ragas_metrics.SentenceTransformer")
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
    r_eval = RagasEvals(mock_generate_response)
    score = r_eval.ragas_answer_relevance(question, answer)
    assert round(score) == 100


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_ragas_context_precision(mock_generate_response):
    question = "What is the name of the largest bone in the human body?"
    retrieved_contexts = ["Retrieved context 1", "Retrieved context 2"]
    mock_generate_response.generate_response.side_effect = ["Yes", "No", "Yes", "No"]

    r_eval = RagasEvals(mock_generate_response)
    score = r_eval.ragas_context_precision(question, retrieved_contexts)

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

    r_eval = RagasEvals(mock_generate_response)
    score = r_eval.ragas_context_recall(question, answer, context)
    assert score == 75


@patch("rag_experiment_accelerator.evaluation.promptflow_quality_metrics.AzureOpenAIModelConfiguration")
def test_promptflow_fluency_evaluator(mock_model_config):
    mock_model_config.return_value = "model_config"
    p_eval = PromptFlowEvals(mock_model_config)

    question = "What is the name of the largest bone in the human body?"
    good_answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."
    bad_answer = "The bone human largest body femur not."

    good_score = p_eval.fluency_evaluator(question, good_answer)
    bad_score = p_eval.fluency_evaluator(question, bad_answer)

    assert good_score == 5
    assert bad_score == 1


@patch("rag_experiment_accelerator.evaluation.promptflow_quality_metrics.AzureOpenAIModelConfiguration")
def test_promptflow_groundedness_evaluator(mock_model_config):
    mock_model_config.return_value = "model_config"
    p_eval = PromptFlowEvals(mock_model_config)

    answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."
    ungrounded_contexts = ["Retrieved context 1", "Retrieved context 2"]
    true_context = 'According to the Cleveland Clinic, "The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average adult male femur is 48 cm (18.9 in) in length and 2.34 cm (0.92 in) in diameter. The average weight among adult males in the United States is 196 lbs (872 N). Therefore, the adult male femur can support roughly 6,000 lbs of compressive force."'
    grounded_contexts = ungrounded_contexts + [true_context]

    low_score = p_eval.groundedness_evaluator(answer, ungrounded_contexts)
    high_score = p_eval.groundedness_evaluator(answer, grounded_contexts)
    assert low_score == 1
    assert high_score == 5


@patch("rag_experiment_accelerator.evaluation.promptflow_quality_metrics.AzureOpenAIModelConfiguration")
def test_promptflow_similarity_evaluator(mock_model_config):
    mock_model_config.return_value = "model_config"
    p_eval = PromptFlowEvals(mock_model_config)

    question = "What is the name of the largest bone in the human body?"
    ground_truth = "The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average length of the femur is 49.5 cm (19.4 inches)."
    good_answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."
    bad_answer = "The largest bone in the human body is the nasal bone."

    good_score = p_eval.similarity_evaluator(question, good_answer, ground_truth)
    bad_score = p_eval.similarity_evaluator(question, bad_answer, ground_truth)
    assert good_score == 5
    assert bad_score == 1


@patch("rag_experiment_accelerator.evaluation.promptflow_quality_metrics.AzureOpenAIModelConfiguration")
def test_promptflow_coherence_evaluator(mock_model_config):
    mock_model_config.return_value = "model_config"
    p_eval = PromptFlowEvals(mock_model_config)

    question = "What is the name of the largest bone in the human body?"
    coherent_answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."
    incoherent_answer = "The largest bile in the human racquet is the tennis ball, also known as the thigh bone."

    good_score = p_eval.coherence_evaluator(question, coherent_answer)
    bad_score = p_eval.coherence_evaluator(question, incoherent_answer)
    assert good_score == 5
    assert bad_score == 1


@patch("rag_experiment_accelerator.evaluation.promptflow_quality_metrics.AzureOpenAIModelConfiguration")
def test_promptflow_relevance_evaluator(mock_model_config):
    mock_model_config.return_value = "model_config"
    p_eval = PromptFlowEvals(mock_model_config)

    question = "What is the name of the largest bone in the human body?"
    relevant_answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."
    irrelevant_answer = "Roger Federer is one of the greatest tennis players of all time."

    good_score = p_eval.relevance_evaluator(question, relevant_answer)
    bad_score = p_eval.relevance_evaluator(question, irrelevant_answer)
    assert good_score == 5
    assert bad_score == 1
