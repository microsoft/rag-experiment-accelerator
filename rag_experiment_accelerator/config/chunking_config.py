from enum import StrEnum
from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


class ChunkingStrategy(StrEnum):
    BASIC = "basic"
    AZURE_DOCUMENT_INTELLIGENCE = "azure-document-intelligence"


@dataclass
class ChunkingConfig(BaseConfig):
    preprocess: bool = False
    chunk_size: int = 512
    overlap: int = 128
    generate_title: bool = False
    generate_summary: bool = False
    override_content_with_summary: bool = False
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.BASIC  # todo: to index?
    # only for azure document intelligence strategy
    azure_document_intelligence_model: str = "prebuilt-read"

    def label_properties(self) -> dict:
        """
        Returns properties used for labeling.
        """
        properties = {
            "p": int(self.preprocess),
            "cs": self.chunk_size,
            "o": self.overlap,
            "t": int(self.generate_title),
            "s": int(self.generate_summary),
            "oc": int(self.override_content_with_summary),
        }

        return properties

    @classmethod
    def from_label_properties(cls, properties: dict) -> "ChunkingConfig":
        """
        Creates ChunkingConfig from the dictionary with properties.
        Reverse of label_properties().
        """

        return ChunkingConfig(
            preprocess=bool(properties["p"]),
            chunk_size=int(properties["cs"]),
            overlap=int(properties["o"]),
            generate_title=bool(properties["t"]),
            generate_summary=bool(properties["s"]),
            override_content_with_summary=bool(properties["os"]),
        )
