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
    ef_construction: int = 400
    ef_search: int = 400
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)

    def label_properties(self) -> dict:
        """
        Returns properties subset used for labeling.
        """
        properties = {
            "idx": self.index_name_prefix,
            "efc": self.ef_construction,
            "efs": self.ef_search,
            "emn": self.embedding_model.model_name.lower(),
            # "sp": self.sampling_percentage,
        }

        properties.update(self.chunking_config.label_properties())

        return properties

    @classmethod
    def from_label_properties(cls, properties: dict) -> "IndexConfig":
        """
        Creates IndexConfig from the dictionary with properties.
        Reverse of label_properties().
        """

        # todo: update validation
        # if len(properties) != 11:
        #     raise (f"Invalid index name [{index_name}]")

        return IndexConfig(
            index_name_prefix=bool(properties["p"]),
            ef_construction=int(properties["cs"]),
            ef_search=int(properties["o"]),
            chunking_config=ChunkingConfig(properties),
            # "emn": self.embedding_model.model_name.lower(),
            # "sp": self.sampling_percentage,
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

        return IndexConfig.from_label_properties(properties)
