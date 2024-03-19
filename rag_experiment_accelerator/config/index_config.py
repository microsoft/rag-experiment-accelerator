from dataclasses import dataclass
from typing import Any

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel


@dataclass
class IndexConfig:
    """A class to hold parameters for each index configured through Config.

    Attributes:
        index_name_prefix (str):
            Prefix to use for the index created in Azure Search.
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

    index_name_prefix: str
    chunk_size: int
    overlap: int
    embedding_model: EmbeddingModel
    ef_construction: int
    ef_search: int

    def index_name(self) -> str:
        """
        Returns index name from the fields.
        Reverse of IndexConfig.from_index_name().
        """
        return (
            f"{self.index_name_prefix}"
            f"_{str(self.chunk_size)}"
            f"_{str(self.overlap)}"
            f"_{str(self.embedding_model.name.lower())}"
            f"_{str(self.ef_construction)}"
            f"_{str(self.ef_search)}"
        )

    @classmethod
    def from_index_name(cls, index_name: str, config: Any) -> "IndexConfig":
        """
        Creates IndexConfig from the index name.
        Reverse of index_name().
        """
        values = index_name.split("_")
        assert len(values) == 6
        return IndexConfig(
            index_name_prefix=values[0],
            chunk_size=int(values[1]),
            overlap=int(values[2]),
            embedding_model=config._find_embedding_model_by_name(values[3]),
            ef_construction=int(values[4]),
            ef_search=int(values[5]),
        )
