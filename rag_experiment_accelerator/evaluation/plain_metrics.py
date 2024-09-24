import evaluate
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

    results = bleu.compute(predictions=predictions, references=references, max_order=2)
    # multiplying by 100 to maintain consistency with previous implementation
    return results["bleu"] * 100


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
            - 'rouge1_f1'
            - 'rouge2_precision'
            - 'rouge2_recall'
            - 'rouge2_f1'
            - 'rougeL_precision'
            - 'rougeL_recall'
            - 'rougeL_f1'
    Returns:
        score: ROUGE score.
    """
    # validate rouge_types to be one of the supported rouge metrics
    supported_rouge_types = {"rouge1", "rouge2", "rougeL"}
    rouge_type, metric_type = rouge_metric_name.split("_")
    if rouge_type not in supported_rouge_types:
        raise ValueError(f"Rouge type '{rouge_type}' is not recognized. "
                         "Supported types are {supported_rouge_types}.")

    if metric_type not in {"precision", "recall", "f1"}:
        raise ValueError(f"Rouge metric type '{rouge_type}' is not recognized. "
                         "Supported metric types are {'precision', 'recall', 'f1'}.")

    scorer = rouge_scorer.RougeScorer(rouge_types=[rouge_type], use_stemmer=True)
    scores = scorer.score(target=ground_truth, prediction=prediction)
    return scores[rouge_type][metric_type]


def levenshtein(str1: str, str2: str) -> int:
    """
    Calculates the Levenshtein distance between two strings and returns the normalized similarity score as a percentage.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        int: The normalized similarity score as a percentage.
    """
    score = distance.Levenshtein.normalized_similarity(str1, str2) * 100
    return score


def jaccard(str1: str, str2: str) -> int:
    """
    Calculates the Jaccard similarity score between two sets of values.

    Args:
        str1 (set): The first set of values.
        str2 (set): The second set of values.

    Returns:
        int: The Jaccard similarity score between the two sets of values, as a percentage.
    """
    score = int(algorithms.jaccard.normalized_similarity(str1, str2) * 100)
    return score


def hamming(str1: str, str2: str) -> int:
    """
    Calculates the Hamming similarity score between two strings.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        int: The Hamming similarity score between the two values, as a percentage.
    """
    score = int(distance.Hamming.normalized_similarity(str1, str2, pad=True) * 100)
    return score


def jaro_winkler(str1: str, str2: str) -> int:
    """
    Calculates the Jaro-Winkler similarity score between two strings.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        int: The Jaro-Winkler similarity score between the two strings, as an integer between 0 and 100.
    """
    score = int(distance.JaroWinkler.normalized_similarity(str1, str2) * 100)
    return score


def lcsseq(str1: str, str2: str) -> int:
    """
    Computes the longest common subsequence (LCS) similarity score between two input strings.

    Args:
        str1 (str): The first input string.
        str2 (str): The second input string.

    Returns:
        int: The LCS similarity score between the two input strings, as a percentage (0-100).
    """
    score = int(distance.LCSseq.normalized_similarity(str1, str2) * 100)
    return score


def lcsstr(str1: str, str2: str) -> int:
    """
    Calculates the longest common substring (LCS) similarity score between two strings.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        int: The LCS similarity score between the two strings, as a percentage (0-100).
    """
    score = int(algorithms.lcsstr.normalized_similarity(str1, str2) * 100)
    return score
