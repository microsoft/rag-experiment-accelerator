from dataclasses import dataclass

from rag_experiment_accelerator.config.base_config import BaseConfig
from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)


@dataclass
class IndexConfig(BaseConfig):
    """A class to hold parameters for each index configured through Config.

    Attributes:
        index_name_prefix (str):
            Prefix to use for the index created in Azure Search.
        preprocess (bool):
            Whether to preprocess the text before indexing.
        chunk_size (int):
            Chunk size for chunking documents.
        overlap (int):
            Overlap size for chunking documents.
        embedding_model (int):
            Embedding model to use for this config.
        ef_construction (int):
            Parameter ef_construction for HNSW index.
        ef_search (int):
            Parameter ef_search for HNSW index.
    """

    index_name_prefix: str = "idx"
    chunking_config: ChunkingConfig = ChunkingConfig()
    embedding_model: EmbeddingModelConfig = EmbeddingModelConfig()
    ef_construction: int = 400
    ef_search: int = 400
