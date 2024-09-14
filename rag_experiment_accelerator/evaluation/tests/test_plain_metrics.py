from unittest.mock import patch

from rag_experiment_accelerator.evaluation.plain_metrics import (
    bleu,
    fuzzy,
    levenshtein,
    jaccard,
    hamming,
    jaro_winkler,
    cosine,
    lcsseq,
    lcsstr,
)


def test_fuzzy():
    value1 = "Room, 2 Double Beds (19th to 25th Floors)"
    value2 = "Two Double Beds - Location Room (19th to 25th Floors)"

    assert int(fuzzy(str1=value1, str2=value2)) == 89
    assert int(fuzzy(str1=value1, str2=value2, match_type="partial_token_set_ratio")) == 100


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


@patch("rag_experiment_accelerator.evaluation.plain_metrics.evaluate.load")
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
