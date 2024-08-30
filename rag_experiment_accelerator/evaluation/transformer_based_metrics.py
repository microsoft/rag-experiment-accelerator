from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# todo: can we remove this hardcoding and name the model in the config file?
metric_type_model_mapping = {
    "bert_all_MiniLM_L6_v2": "all-MiniLM-L6-v2",
    "bert_base_nli_mean_tokens": "bert-base-nli-mean-tokens",
    "bert_large_nli_mean_tokens": "bert-large-nli-mean-tokens",
    "bert_large_nli_stsb_mean_tokens": "bert-large-nli-stsb-mean-tokens",
    "bert_distilbert_base_nli_stsb_mean_tokens": "distilbert-base-nli-stsb-mean-tokens",
    "bert_paraphrase_multilingual_MiniLM_L12_v2": "paraphrase-multilingual-MiniLM-L12-v2",
}


def compare_semantic_document_values(doc1, doc2, model_type):
    """
    Compares the semantic values of two documents and returns the percentage of differences.

    Args:
        doc1 (str): The first document to compare.
        doc2 (str): The second document to compare.
        model_type (SentenceTransformer): The SentenceTransformer model to use for comparison.

    Returns:
        int: The percentage of differences between the two documents.
    """
    differences = semantic_compare_values(doc1, doc2, model_type)

    return int(sum(differences) / len(differences))


def semantic_compare_values(
    value1: str,
    value2: str,
    model_type: SentenceTransformer,
) -> list[float]:
    """
    Computes the semantic similarity between two values using a pre-trained SentenceTransformer model.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        model_type (SentenceTransformer): The pre-trained SentenceTransformer model to use for encoding the values.

    Returns:
        A list of the similarity scores.
    """
    embedding1 = model_type.encode([str(value1)])
    embedding2 = model_type.encode([str(value2)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    return [similarity_score * 100]


def compute_transformer_based_score(
    actual,
    expected,
    metric_type,
):
    if metric_type not in metric_type_model_mapping:
        raise KeyError(f"Invalid metric type: {metric_type}")

    transformer = SentenceTransformer(
        f"sentence-transformers/{metric_type_model_mapping[metric_type]}"
    )
    return compare_semantic_document_values(actual, expected, transformer)
