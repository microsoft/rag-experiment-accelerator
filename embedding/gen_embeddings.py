import os

import openai
from sentence_transformers import SentenceTransformer


from typing import List
from sentence_transformers import SentenceTransformer


size_model_mapping = {
    384: 'all-MiniLM-L6-v2',
    768: 'all-mpnet-base-v2',
    1024: 'bert-large-nli-mean-tokens',
}


def generate_embedding(size: int, chunk: str) -> List[float]:
    """
    Generates an embedding for a given text chunk using a pre-trained transformer model.

    Args:
        size (int): The size of the transformer model to use. Must be one of 384, 768, 1024 or 1536.
        chunk (str): The text chunk to generate an embedding for.

    Returns:
        List[float]: A list of floats representing the embedding for the given text chunk.
    """
    if size == 1536:
        embedding = openai.Embedding.create(input=[chunk],
                                            engine=os.getenv('OPENAI_EMBEDDING_DEPLOYED_MODEL'))['data'][0]['embedding']
        return [embedding]

    if size in size_model_mapping:
        model = SentenceTransformer(size_model_mapping[size])
        return model.encode([str(chunk)]).tolist()

    # todo: log error and/or handle the default setup
