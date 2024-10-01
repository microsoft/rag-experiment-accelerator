import evaluate
<<<<<<< HEAD
from rapidfuzz import fuzz
from rapidfuzz import distance
from textdistance import algorithms
from rouge_score import rouge_scorer


# https://huggingface.co/spaces/evaluate-metric/bleu
def bleu(predictions: list[str], references: list[str]) -> float:
    """
    Computes the BLEU score between a list of candidate translations and a list of reference translations.

    Args:
        predictions (list): A list of candidate translations.
        references (list): A list of reference translations.

    Returns:
        float: The BLEU score between the candidate and reference translations.
    """
    bleu = evaluate.load("bleu")

    # Match length of predictions and references and check they are both lists of strings
    if len(predictions) != len(references) or not all(isinstance(x, str) for x in predictions) or not all(
        isinstance(x, list) for x in references
    ):
        raise ValueError("Predictions and references must be lists of strings with the same length.")

=======
import textdistance
from fuzzywuzzy import fuzz

algorithms = textdistance.algorithms


# https://huggingface.co/spaces/evaluate-metric/bleu
def bleu(predictions, references):
    bleu = evaluate.load("bleu")

>>>>>>> main
    results = bleu.compute(predictions=predictions, references=references, max_order=2)
    # multiplying by 100 to maintain consistency with previous implementation
    return results["bleu"] * 100


<<<<<<< HEAD
def fuzzy_score(str1: str, str2: str, match_type: str = "token_set_ratio") -> float:
    """
    Compares two strings using fuzzy string matching and returns a similarity score.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
        match_type (str): The type of fuzzy string matching to use. Options include:
            - 'ratio'
            - 'token_set_ratio'
            - 'token_sort_ratio'
            - 'partial_ratio'
            - 'partial_token_sort_ratio'
            - 'partial_token_set_ratio'
            - 'WRatio'
            - 'QRatio'

    Returns:
        A similarity score.

    Raises:
        ValueError: If the match type is not recognized.
    """
    # validate match_type to be one of the supported fuzzy matching functions
    supported_match_types = {"ratio",
                             "token_set_ratio",
                             "token_sort_ratio",
                             "partial_ratio",
                             "partial_token_sort_ratio",
                             "partial_token_set_ratio",
                             "WRatio",
                             "QRatio"}
    if match_type not in supported_match_types:
        raise ValueError(f"Match type '{match_type}' is not recognized.")

    # get the fuzzy matching function based on the match_type
    fuzzy_match_fn = getattr(fuzz, match_type)
    similarity_score = fuzzy_match_fn(str1, str2)
    return similarity_score


def rouge_score(ground_truth: str, prediction: str, rouge_metric_name: str) -> float:
    """
    Calculates the ROUGE scores (rouge1, rouge2, rougeL) between two strings - ground truth and prediction.

    Args:
        ground_truth: reference string to compare
        prediction: string that is an output of a model, a system or a generating process
        rouge_metric_name: list of rouge metrics to use for evaluation. Options include:
            - 'rouge1_precision'
            - 'rouge1_recall'
            - 'rouge1_fmeasure'
            - 'rouge2_precision'
            - 'rouge2_recall'
            - 'rouge2_fmeasure'
            - 'rougeL_precision'
            - 'rougeL_recall'
            - 'rougeL_fmeasure'
    Returns:
        score: ROUGE score.
    """
    # validate rouge_types to be one of the supported rouge metrics
    supported_rouge_types = {"rouge1", "rouge2", "rougeL"}
    rouge_type, metric_type = rouge_metric_name.split("_")
    if rouge_type not in supported_rouge_types:
        raise ValueError(f"Rouge type '{rouge_type}' is not recognized. "
                         "Supported types are {supported_rouge_types}.")

    if metric_type not in {"precision", "recall", "fmeasure"}:
        raise ValueError(f"Rouge metric type '{rouge_type}' is not recognized. "
                         "Supported metric types are {'precision', 'recall', 'fmeasure'}.")

    scorer = rouge_scorer.RougeScorer(rouge_types=[rouge_type], use_stemmer=True)
    scores = scorer.score(target=ground_truth, prediction=prediction)
    return getattr(scores[rouge_type], metric_type) * 100


def levenshtein(str1: str, str2: str) -> int:
=======
def fuzzy(doc1, doc2):
    """
    Calculates the fuzzy score between two documents.

    Parameters:
        doc1 (str): The first document to compare.
        doc2 (str): The second document to compare.

    Returns:
        int: The fuzzy score between the two documents.
    """
    differences = fuzzy_compare_values(doc1, doc2)

    return int(sum(differences) / len(differences))


def fuzzy_compare_values(value1, value2) -> list[float]:
    """
    Compares two values using fuzzy string matching and appends the similarity score to a list of differences.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.

    Returns:
        A list of the similarity scores.
    """
    similarity_score = [fuzz.token_set_ratio(value1, value2)]
    return similarity_score


def levenshtein(value1, value2):
>>>>>>> main
    """
    Calculates the Levenshtein distance between two strings and returns the normalized similarity score as a percentage.

    Args:
<<<<<<< HEAD
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
=======
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.
>>>>>>> main

    Returns:
        int: The normalized similarity score as a percentage.
    """
<<<<<<< HEAD
    score = distance.Levenshtein.normalized_similarity(str1, str2) * 100
    return score


def jaccard(str1: str, str2: str) -> int:
=======
    score = int(algorithms.levenshtein.normalized_similarity(value1, value2) * 100)
    return score


def jaccard(value1, value2):
>>>>>>> main
    """
    Calculates the Jaccard similarity score between two sets of values.

    Args:
<<<<<<< HEAD
        str1 (set): The first set of values.
        str2 (set): The second set of values.
=======
        value1 (set): The first set of values.
        value2 (set): The second set of values.
>>>>>>> main

    Returns:
        int: The Jaccard similarity score between the two sets of values, as a percentage.
    """
<<<<<<< HEAD
    score = int(algorithms.jaccard.normalized_similarity(str1, str2) * 100)
    return score


def hamming(str1: str, str2: str) -> int:
    """
    Calculates the Hamming similarity score between two strings.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
=======
    score = int(algorithms.jaccard.normalized_similarity(value1, value2) * 100)
    return score


def hamming(value1, value2):
    """
    Calculates the Hamming similarity score between two values.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
>>>>>>> main

    Returns:
        int: The Hamming similarity score between the two values, as a percentage.
    """
<<<<<<< HEAD
    score = int(distance.Hamming.normalized_similarity(str1, str2, pad=True) * 100)
    return score


def jaro_winkler(str1: str, str2: str) -> int:
=======
    score = int(algorithms.hamming.normalized_similarity(value1, value2) * 100)
    return score


def jaro_winkler(value1, value2):
>>>>>>> main
    """
    Calculates the Jaro-Winkler similarity score between two strings.

    Args:
<<<<<<< HEAD
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
=======
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.
>>>>>>> main

    Returns:
        int: The Jaro-Winkler similarity score between the two strings, as an integer between 0 and 100.
    """
<<<<<<< HEAD
    score = int(distance.JaroWinkler.normalized_similarity(str1, str2) * 100)
    return score


def cosine_ochiai(str1: str, str2: str) -> float:
    """
    Calculates the cosine similarity (Ochiai coefficient) between two strings
    using token-frequency vectors
    https://en.wikipedia.org/wiki/Cosine_similarity.
    Args:
        str1 (str): The first string.
        str2 (str): The second string.
    Returns:
        float: The cosine similarity score between the two strings, as a percentage.
    """
    score = int(algorithms.cosine.normalized_similarity(str1, str2) * 100)
    return score


def lcsseq(str1: str, str2: str) -> int:
=======
    score = int(algorithms.jaro_winkler.normalized_similarity(value1, value2) * 100)
    return score


def cosine(value1, value2):
    """
    Calculates the cosine similarity (Ochiai coefficient) between two strings
    using token-frequency vectors

    https://en.wikipedia.org/wiki/Cosine_similarity.

    Args:
        value1 (list): The first vector.
        value2 (list): The second vector.

    Returns:
        int: The cosine similarity score between the two vectors, as a percentage.
    """
    score = int(algorithms.cosine.normalized_similarity(value1, value2) * 100)
    return score


def lcsseq(value1, value2):
>>>>>>> main
    """
    Computes the longest common subsequence (LCS) similarity score between two input strings.

    Args:
<<<<<<< HEAD
        str1 (str): The first input string.
        str2 (str): The second input string.
=======
        value1 (str): The first input string.
        value2 (str): The second input string.
>>>>>>> main

    Returns:
        int: The LCS similarity score between the two input strings, as a percentage (0-100).
    """
<<<<<<< HEAD
    score = int(distance.LCSseq.normalized_similarity(str1, str2) * 100)
    return score


def lcsstr(str1: str, str2: str) -> int:
=======
    score = int(algorithms.lcsseq.normalized_similarity(value1, value2) * 100)
    return score


def lcsstr(value1, value2):
>>>>>>> main
    """
    Calculates the longest common substring (LCS) similarity score between two strings.

    Args:
<<<<<<< HEAD
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
=======
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.
>>>>>>> main

    Returns:
        int: The LCS similarity score between the two strings, as a percentage (0-100).
    """
<<<<<<< HEAD
    score = int(algorithms.lcsstr.normalized_similarity(str1, str2) * 100)
=======
    score = int(algorithms.lcsstr.normalized_similarity(value1, value2) * 100)
>>>>>>> main
    return score
