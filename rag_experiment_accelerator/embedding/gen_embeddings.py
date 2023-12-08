from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from rag_experiment_accelerator.config.config import OpenAICredentials

size_model_mapping = {
    384: "all-MiniLM-L6-v2",
    768: "all-mpnet-base-v2",
    1024: "bert-large-nli-mean-tokens",
}


# If we were about to choose model by the name, code would be cleaner.
# We could give user an option to choose the model either by the name or by the size, whichever is specified.
def generate_embedding(size: int, chunk: str, model_name: str, openai_creds: OpenAICredentials) -> list[float]:
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
        if openai_creds.OPENAI_API_TYPE == 'azure':

            client = AzureOpenAI(
                azure_endpoint=openai_creds.OPENAI_ENDPOINT, 
                api_key=openai_creds.OPENAI_API_KEY,  
                api_version=openai_creds.OPENAI_API_VERSION
            )
        else:
            client = OpenAI(
                api_key=openai_creds.OPENAI_API_KEY,  
            )

        response = client.embeddings.create(
            input=chunk,
            model=model_name
        )

        embedding = response.data[0].embedding
        return [embedding]

    if size in size_model_mapping:
        model = SentenceTransformer(size_model_mapping[size])
        return model.encode([str(chunk)]).tolist()

    # todo: log error and/or handle the default setup
