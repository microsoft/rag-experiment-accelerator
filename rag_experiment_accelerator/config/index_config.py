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
    overlap_size: int
    embedding_model: EmbeddingModel
    ef_construction: int
    ef_search: int
    sampling_percentage: int = 0
    generate_title: bool = False
    generate_summary: bool = False
    override_content_with_summary: bool = False

    def index_name(self) -> str:
        """
        Returns index name from the fields.
        Reverse of IndexConfig.from_index_name().
        """
        index_name = (
            f"{self.index_name_prefix}"
            f"_cs-{str(self.chunk_size)}"
            f"_o-{str(self.overlap_size)}"
            f"_efc-{str(self.ef_construction)}"
            f"_efs-{str(self.ef_search)}"
            f"_sp-{str(self.sampling_percentage)}"
            f"_t-{int(self.generate_title)}"
            f"_s-{int(self.generate_summary)}"
            f"_oc-{int(self.override_content_with_summary)}"
            f"_{str(self.embedding_model.name.lower())}"
        )

        return index_name

    @classmethod
    def from_index_name(cls, index_name: str, config: Any) -> "IndexConfig":
        """
        Creates IndexConfig from the index name.
        Reverse of index_name().
        """
        values = index_name.split("_")
        if len(values) != 10:
            raise (f"Invalid index name [{index_name}]")

        return IndexConfig(
            index_name_prefix=values[0],
            chunk_size=int(values[1].split("-")[1].strip()),
            overlap_size=int(values[2].split("-")[1].strip()),
            ef_construction=int(values[3].split("-")[1].strip()),
            ef_search=int(values[4].split("-")[1].strip()),
            sampling_percentage=int(values[5].split("-")[1].strip()),
            generate_title=bool(int(values[6].split("-")[1].strip())),
            generate_summary=bool(int(values[7].split("-")[1].strip())),
            override_content_with_summary=bool(int(values[8].split("-")[1].strip())),
            embedding_model=config._find_embedding_model_by_name(values[9].strip()),
        )
