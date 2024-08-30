import evaluate
import textdistance
from fuzzywuzzy import fuzz

algorithms = textdistance.algorithms


# https://huggingface.co/spaces/evaluate-metric/bleu
def bleu(predictions, references):
    bleu = evaluate.load("bleu")

    results = bleu.compute(predictions=predictions, references=references, max_order=2)
    # multiplying by 100 to maintain consistency with previous implementation
    return results["bleu"] * 100


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
    """
    Calculates the Levenshtein distance between two strings and returns the normalized similarity score as a percentage.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The normalized similarity score as a percentage.
    """
    score = int(algorithms.levenshtein.normalized_similarity(value1, value2) * 100)
    return score


def jaccard(value1, value2):
    """
    Calculates the Jaccard similarity score between two sets of values.

    Args:
        value1 (set): The first set of values.
        value2 (set): The second set of values.

    Returns:
        int: The Jaccard similarity score between the two sets of values, as a percentage.
    """
    score = int(algorithms.jaccard.normalized_similarity(value1, value2) * 100)
    return score


def hamming(value1, value2):
    """
    Calculates the Hamming similarity score between two values.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.

    Returns:
        int: The Hamming similarity score between the two values, as a percentage.
    """
    score = int(algorithms.hamming.normalized_similarity(value1, value2) * 100)
    return score


def jaro_winkler(value1, value2):
    """
    Calculates the Jaro-Winkler similarity score between two strings.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The Jaro-Winkler similarity score between the two strings, as an integer between 0 and 100.
    """
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
    """
    Computes the longest common subsequence (LCS) similarity score between two input strings.

    Args:
        value1 (str): The first input string.
        value2 (str): The second input string.

    Returns:
        int: The LCS similarity score between the two input strings, as a percentage (0-100).
    """
    score = int(algorithms.lcsseq.normalized_similarity(value1, value2) * 100)
    return score


def lcsstr(value1, value2):
    """
    Calculates the longest common substring (LCS) similarity score between two strings.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The LCS similarity score between the two strings, as a percentage (0-100).
    """
    score = int(algorithms.lcsstr.normalized_similarity(value1, value2) * 100)
    return score
