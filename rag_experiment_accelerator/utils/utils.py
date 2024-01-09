def get_index_name(prefix: str, chunk_size: int, overlap: int, embedding_model_name: str, ef_construction: int, ef_search: int) -> str:
    return f"{prefix}-{chunk_size}-{overlap}-{embedding_model_name.lower()}-{ef_construction}-{ef_search}"
