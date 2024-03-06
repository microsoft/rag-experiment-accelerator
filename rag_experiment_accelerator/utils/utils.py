def get_index_name(
    prefix: str,
    chunk_size: int,
    overlap: int,
    embedding_model_name: str,
    ef_construction: int,
    ef_search: int,
    sample_data: bool,
    sample_percentage: int,
) -> str:
    """
    Generates an index name based on the provided parameters.

    Args:
        prefix (str): The prefix for the index name.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between chunks.
        embedding_model_name (str): The name of the embedding model.
        ef_construction (int): The construction parameter for the index.
        ef_search (int): The search parameter for the index.
        sample_data (bool): Whether to sample the data.
        sample_percentage (int): The percentage of data to sample.

    Returns:
        str: The generated index name.
    """

    index_name = f"{prefix}-{chunk_size}-{overlap}-{embedding_model_name.lower()}-{ef_construction}-{ef_search}"
    if sample_data:
        index_name = index_name + f"_sampled_{sample_percentage}"

    return index_name
