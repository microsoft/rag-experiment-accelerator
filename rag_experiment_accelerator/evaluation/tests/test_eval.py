from unittest.mock import MagicMock, patch

import numpy as np

from rag_experiment_accelerator.evaluation.eval import (
    lower,
    remove_spaces,
    bleu,
    fuzzy,
    compare_semantic_document_values,
    levenshtein,
    jaccard,
    hamming,
    jaro_winkler,
    cosine,
    lcsseq,
    lcsstr,
    llm_answer_relevance,
    llm_context_precision,
    llm_context_recall,
)


def test_lower():
    text = "UPPER CASE input text"
    expected = "upper case input text"

    assert lower(text) == expected


def test_remove_spaces():
    text = "  leading and trailing spaces   "
    expected = "leading and trailing spaces"

    assert remove_spaces(text) == expected


def test_fuzzy():
    value1 = "Room, 2 Double Beds (19th to 25th Floors)"
    value2 = "Two Double Beds - Location Room (19th to 25th Floors)"

    assert fuzzy(value1, value2) == 97


def test_compare_semantic_document_values():
    mock_sentence_transformer = MagicMock()
    embeddings1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.7]])
    embeddings2 = np.array([[0.1, 0.3, 0.4, 0.5, 0.6]])

    mock_sentence_transformer.encode.side_effect = [embeddings1, embeddings2]

    value1 = "value1"
    value2 = "value2"

    assert (
        compare_semantic_document_values(value1, value2, mock_sentence_transformer)
        == 97
    )


def test_levenshtein():
    value1 = "party"
    value2 = "park"

    assert levenshtein(value1, value2) == 60


def test_jaccard():
    value1 = ["cat", "dog", "hippo", "monkey"]
    value2 = ["monkey", "rhino", "ostrich", "salmon"]

    assert jaccard(value1, value2) == 14


def test_hamming():
    value1 = "1011101"
    value2 = "1011011"

    assert hamming(value1, value2) == 71


def test_jaro_winkler():
    value1 = "crate"
    value2 = "trace"

    assert jaro_winkler(value1, value2) == 73


def test_cosine():
    value1 = "Soup is a primarily liquid food, generally served warm or hot (but may be cool or cold), that is made by combining ingredients of meat or vegetables with stock, juice, water, or another liquid. "
    value2 = "Noodles are a staple food in many cultures. They are made from unleavened dough which is stretched, extruded, or rolled flat and cut into one of a variety of shapes."

    assert cosine(value1, value2) == 81


def test_lcsseq():
    value1 = "The fox jumped over the high fence"
    value2 = "The quick brown fox jumped over the fence."

    assert lcsseq(value1, value2) == 69


def test_lcsstr():
    value1 = "The fox jumped over the high fence"
    value2 = "The quick brown fox jumped over the fence."

    assert lcsstr(value1, value2) == 50


@patch("rag_experiment_accelerator.evaluation.eval.evaluate.load")
def test_bleu(mock_evaluate_load):
    mock_evaluate_load.return_value.compute.return_value = {"bleu": 0.5}
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


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
@patch("rag_experiment_accelerator.evaluation.eval.SentenceTransformer")
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
    mock_generate_response.generate_response.return_value = "Yes"
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

    score = llm_context_precision(mock_generate_response, question, context)
    assert score == 100


@patch("rag_experiment_accelerator.evaluation.eval.ResponseGenerator")
def test_llm_context_recall(mock_generate_response):
    mock_generate_response.generate_response.return_value = (
        '"Attributed": "1"   "Attributed": "1"   "Attributed": "1"   "Attributed": "0"'
    )
    question = "What is the name of the largest bone in the human body?"
    context = 'According to the Cleveland Clinic, "The femur is the largest and strongest bone in the human body. It can support as much as 30 times the weight of your body. The average adult male femur is 48 cm (18.9 in) in length and 2.34 cm (0.92 in) in diameter. The average weight among adult males in the United States is 196 lbs (872 N). Therefore, the adult male femur can support roughly 6,000 lbs of compressive force."'
    answer = "The largest bone in the human body is the femur, also known as the thigh bone. It is about 19.4 inches (49.5 cm) long on average and can support up to 30 times the weight of a person’s body."

    score = llm_context_recall(mock_generate_response, question, answer, context, 5)
    assert score == 75
