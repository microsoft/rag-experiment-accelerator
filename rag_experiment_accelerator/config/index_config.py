from dataclasses import dataclass, field

from rag_experiment_accelerator.config.base_config import BaseConfig
from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)
from rag_experiment_accelerator.config.sampling_config import SamplingConfig


@dataclass
class IndexConfig(BaseConfig):
    """A class to hold parameters for each index configured through Config.

    Attributes:
        index_name_prefix (str):
            Prefix to use for the index created in Azure Search.
        ef_construction (int):
            Parameter ef_construction for HNSW index.
        ef_search (int):
            Parameter ef_search for HNSW index.
        chunking (ChunkingConfig):
            Configuration for chunking documents.
        embedding_model (EmbeddingModelConfig):
            Configuration for the embedding model.
        sampling (SamplingConfig):
            Configuration for sampling documents.
    """

    index_name_prefix: str = "idx"
    ef_construction: int = 400
    ef_search: int = 400
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def label_properties(self) -> dict:
        """
        Returns properties subset used for labeling.
        """
        properties = {
            "idx": self.index_name_prefix,
            "efc": self.ef_construction,
            "efs": self.ef_search,
            "emn": self.embedding_model.model_name.lower(),
            "sp": self.sampling.percentage,
        }

        properties.update(self.chunking.label_properties())

        return properties

    @classmethod
    def from_label_properties(cls, properties: dict) -> "IndexConfig":
        """
        Creates IndexConfig from the dictionary with properties.
        Reverse of label_properties().
        """

        return IndexConfig(
            index_name_prefix=properties["idx"],
            ef_construction=int(properties["efc"]),
            ef_search=int(properties["efs"]),
            chunking=ChunkingConfig.from_label_properties(properties),
            embedding_model=EmbeddingModelConfig(model_name=properties["emn"]),
            sampling=SamplingConfig(percentage=properties["sp"]),
        )

    def index_name(self) -> str:
        """
        Returns index name from the fields.
        Reverse of IndexConfig.from_index_name().
        """
        index_name = "_".join(
            [f"{key}-{value}" for (key, value) in self.label_properties().items()]
        )
        if index_name.startswith("_") or index_name.startswith("-"):
            index_name = "i" + index_name

        index_name = index_name[:127]

        return index_name

    @classmethod
    def from_index_name(cls, index_name: str) -> "IndexConfig":
        """
        Creates IndexConfig from the index name.
        Reverse of index_name().
        """

        key_values = [kv.split("-") for kv in index_name.split("_")]
        properties = {kv[0]: kv[1].strip() for kv in key_values}

        try:
            index_config = IndexConfig.from_label_properties(properties)
        except Exception as e:
            raise ValueError(f"Invalid index name [{index_name}]. {e}")

        return index_config
