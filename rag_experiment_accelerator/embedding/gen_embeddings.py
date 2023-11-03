import openai
from sentence_transformers import SentenceTransformer

size_model_mapping = {
    384: "all-MiniLM-L6-v2",
    768: "all-mpnet-base-v2",
    1024: "bert-large-nli-mean-tokens",
}


# If we were about to choose model by the name, code would be cleaner.
# We could give user an option to choose the model either by the name or by the size, whichever is specified.
def generate_embedding(size: int, chunk: str, model_name: str) -> list[float]:
    """
    Generates an embedding for a given text chunk using a pre-trained transformer model.

    Args:
        size (int): The size of the transformer model to use. Must be one of 384, 768, 1024 or 1536.
        chunk (str): The text chunk to generate an embedding for.
        model_name (str): Name of the model used to generate the embedding.

    Returns:
        list[float]: A list of floats representing the embedding for the given text chunk.
    """
    if size == 1536:
        params = {
            "input": [chunk],
        }
        if openai.api_type == "azure":
            params["engine"] = model_name
        else:
            params["model"] = model_name

        embedding = openai.Embedding.create(**params)["data"][0]["embedding"]
        return [embedding]

    if size in size_model_mapping:
        model = SentenceTransformer(size_model_mapping[size])
        return model.encode([str(chunk)]).tolist()

    # todo: log error and/or handle the default setup
