
from sentence_transformers import SentenceTransformer

from typing import List
from sentence_transformers import SentenceTransformer

def generate_embedding(size: int, chunk: str) -> List[float]:
    """
    Generates an embedding for a given text chunk using a pre-trained transformer model.

    Args:
        size (int): The size of the transformer model to use. Must be one of 384, 768, or 1024.
        chunk (str): The text chunk to generate an embedding for.

    Returns:
        List[float]: A list of floats representing the embedding for the given text chunk.
    """
    if size == 384:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif size == 768:
        model = SentenceTransformer('all-mpnet-base-v2')
    elif size == 1024:
        model = SentenceTransformer('bert-large-nli-mean-tokens')
    else:
        model = ""
    return model.encode([str(chunk)]).tolist()
