import evaluate
from rapidfuzz import fuzz
from rapidfuzz import distance as d
from textdistance import algorithms as alg


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

    results = bleu.compute(predictions=predictions, references=references, max_order=2)
    # multiplying by 100 to maintain consistency with previous implementation
    return results["bleu"] * 100


def fuzzy(str1: str, str2: str, match_type: str = "token_set_ratio") -> float:
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
        A list of the similarity scores.

    Raises:
        ValueError: If the match type is not recognized.
    """
    try:
        fuzzy_match_fn = getattr(fuzz, match_type)
    except AttributeError:
        raise ValueError(f"Match type '{match_type}' is not recognized.")

    similarity_score = fuzzy_match_fn(str1, str2)
    return similarity_score


def levenshtein(str1: str, str2: str) -> int:
    """
    Calculates the Levenshtein distance between two strings and returns the normalized similarity score as a percentage.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        int: The normalized similarity score as a percentage.
    """
    score = d.Levenshtein.normalized_similarity(str1, str2) * 100
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
    score = int(alg.jaccard.normalized_similarity(str1, str2) * 100)
    return score


def hamming(str1: str, str2: str) -> int:
    """
    Calculates the Hamming similarity score between two values.

    Args:
        str1 (str): The first value to compare.
        str2 (str): The second value to compare.

    Returns:
        int: The Hamming similarity score between the two values, as a percentage.
    """
    score = int(d.Hamming.normalized_similarity(str1, str2) * 100)
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
    score = int(d.JaroWinkler.normalized_similarity(str1, str2) * 100)
    return score


def cosine(str1: str, str2: str) -> int:
    """
    Calculates the cosine similarity (Ochiai coefficient) between two strings
    using token-frequency vectors

    https://en.wikipedia.org/wiki/Cosine_similarity.

    Args:
        str1 (list): The first vector.
        str2 (list): The second vector.

    Returns:
        int: The cosine similarity score between the two vectors, as a percentage.
    """
    score = int(alg.cosine.normalized_similarity(str1, str2) * 100)
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
    score = int(d.LCSseq.normalized_similarity(str1, str2) * 100)
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
    score = int(alg.lcsstr.normalized_similarity(str1, str2) * 100)
    return score
