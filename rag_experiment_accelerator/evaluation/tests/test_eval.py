from unittest.mock import patch

from rag_experiment_accelerator.evaluation.eval import (bleu,
                                                        llm_answer_relevance,
                                                        llm_context_precision)


def test_bleu():
    predictions = [
        "Transformers Transformers are fast plus efficient",
        "Good Morning",
        "I am waiting for new Transformers",
    ]
    references = [
        [
            "HuggingFace Transformers are quick, efficient and awesome",
            "Transformers are awesome because they are fast to execute",
        ],
        ["Good Morning Transformers", "Morning Transformers"],
        [
            "People are eagerly waiting for new Transformer models",
            "People are very excited about new Transformers",
        ],
    ]
    score = bleu(predictions, references)
    assert round(score) == 50


@patch(
    "rag_experiment_accelerator.llm.response_generator.ResponseGenerator.generate_response"
)
def test_llm_answer_relevance(mock_generate_response):
    mock_generate_response.return_value = (
        "What is the name of the largest bone in the human body?"
    )

    question = "What is the name of the largest bone in the human body?"
    answer = (
        (
            "The largest bone in the human body is the femur, also known as"
            " the thigh bone. It is about 19.4 inches (49.5 cm) long on"
            " average and can support up to 30 times the weight of a person’s"
            " body."
        ),
    )
    score = llm_answer_relevance(question, answer)
    assert round(score) == 100


@patch(
    "rag_experiment_accelerator.llm.response_generator.ResponseGenerator.generate_response"
)
def test_llm_context_precision(mock_generate_response):
    mock_generate_response.return_value = "Yes"
    question = "What is the name of the largest bone in the human body?"
    context = (
        'According to the Cleveland Clinic, "The femur is the largest and'
        " strongest bone in the human body. It can support as much as 30 times"
        " the weight of your body. The average adult male femur is 48 cm (18.9"
        " in) in length and 2.34 cm (0.92 in) in diameter. The average weight"
        " among adult males in the United States is 196 lbs (872 N)."
        " Therefore, the adult male femur can support roughly 6,000 lbs of"
        ' compressive force."'
    )

    score = llm_context_precision(question, context)
    assert score == 100
