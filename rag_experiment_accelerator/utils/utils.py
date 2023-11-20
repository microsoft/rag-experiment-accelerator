from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

def get_index_name(prefix: str, chunk_size: int, overlap: int, embedding_model_name: str, ef_construction: str, ef_search: str):
    index_name = f"{prefix}-{chunk_size}-{overlap}-{embedding_model_name}-{ef_construction}-{ef_search}"
    return index_name.lower()
