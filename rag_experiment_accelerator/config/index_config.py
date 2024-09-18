from dataclasses import dataclass, field
from enum import StrEnum

from rag_experiment_accelerator.config.base_config import BaseConfig
from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)
from rag_experiment_accelerator.config.sampling_config import SamplingConfig


class IndexKey(StrEnum):
    PREFIX = "idx"
    EF_CONSTRUCTION = "efc"
    EF_SEARCH = "efs"
    EMBEDDING_MODEL_NAME = "em"
    DIMENSION = "d"
    SAMPLING_PERCENTAGE = "sp"
    PREPROCESS = "p"
    CHUNK_SIZE = "cs"
    CHUNKING_STRATEGY = "st"
    OVERLAP_SIZE = "o"
    GENERATE_TITLE = "t"
    GENERATE_SUMMARY = "s"
    OVERRIDE_CONTENT_WITH_SUMMARY = "oc"


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

    def __label_properties(self) -> dict:
        """
        Returns properties subset used for labeling.
        """
        properties = {
            IndexKey.PREFIX: self.index_name_prefix,
            IndexKey.EF_CONSTRUCTION: self.ef_construction,
            IndexKey.EF_SEARCH: self.ef_search,
            IndexKey.EMBEDDING_MODEL_NAME: self.embedding_model.model_name.lower(),
            IndexKey.SAMPLING_PERCENTAGE: self.sampling.percentage,
            IndexKey.PREPROCESS: int(self.chunking.preprocess),
            IndexKey.CHUNK_SIZE: self.chunking.chunk_size,
            IndexKey.CHUNKING_STRATEGY: self.chunking.chunking_strategy,
            IndexKey.OVERLAP_SIZE: self.chunking.overlap_size,
            IndexKey.GENERATE_TITLE: int(self.chunking.generate_title),
            IndexKey.GENERATE_SUMMARY: int(self.chunking.generate_summary),
            IndexKey.OVERRIDE_CONTENT_WITH_SUMMARY: int(
                self.chunking.override_content_with_summary
            ),
        }

        if self.embedding_model.dimension:
            properties[IndexKey.DIMENSION] = self.embedding_model.dimension

        return properties

    @classmethod
    def __from_label_properties(cls, properties: dict) -> "IndexConfig":
        """
        Creates IndexConfig from the dictionary with properties.
        Reverse of __label_properties().
        """

        return IndexConfig(
            index_name_prefix=properties[IndexKey.PREFIX],
            ef_construction=int(properties[IndexKey.EF_CONSTRUCTION]),
            ef_search=int(properties[IndexKey.EF_SEARCH]),
            chunking=ChunkingConfig(
                preprocess=bool(properties[IndexKey.PREPROCESS]),
                chunk_size=int(properties[IndexKey.CHUNK_SIZE]),
                chunking_strategy=properties[IndexKey.CHUNKING_STRATEGY],
                overlap_size=int(properties[IndexKey.OVERLAP_SIZE]),
                generate_title=bool(int(properties[IndexKey.GENERATE_TITLE])),
                generate_summary=bool(int(properties[IndexKey.GENERATE_SUMMARY])),
                override_content_with_summary=bool(
                    properties[IndexKey.OVERRIDE_CONTENT_WITH_SUMMARY]
                ),
            ),
            embedding_model=EmbeddingModelConfig(
                model_name=properties[IndexKey.EMBEDDING_MODEL_NAME],
                dimension=int(properties[IndexKey.DIMENSION])
                if IndexKey.DIMENSION in properties
                else None,
            ),
            sampling=SamplingConfig(
                percentage=properties[IndexKey.SAMPLING_PERCENTAGE]
            ),
        )

    def index_name(self) -> str:
        """
        Returns index name from the fields.
        Reverse of IndexConfig.from_index_name().
        """
        index_name = "_".join(
            [f"{key}-{value}" for (key, value) in self.__label_properties().items()]
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

        key_values = [kv.split("-", 1) for kv in index_name.split("_")]
        properties = {kv[0]: kv[1].strip() for kv in key_values}

        try:
            index_config = IndexConfig.__from_label_properties(properties)
        except Exception as e:
            raise ValueError(f"Invalid index name [{index_name}]. {e}")

        return index_config
