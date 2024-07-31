from dataclasses import dataclass, field

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
        chunking_config (ChunkingConfig):
            Configuration for chunking documents.
        embedding_model (EmbeddingModelConfig):
            Configuration for the embedding model.
        ef_construction (int):
            Parameter ef_construction for HNSW index.
        ef_search (int):
            Parameter ef_search for HNSW index.
    """

    index_name_prefix: str = "idx"
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    ef_construction: int = 400
    ef_search: int = 400

    def __pre_init__(self):
        super().__init__()
