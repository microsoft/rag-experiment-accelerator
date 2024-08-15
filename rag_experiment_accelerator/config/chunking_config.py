from enum import StrEnum
from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


class ChunkingStrategy(StrEnum):
    BASIC = "basic"
    AZURE_DOCUMENT_INTELLIGENCE = "azure-document-intelligence"

    def __repr__(self) -> str:
        return f'"{self.value}"'


@dataclass
class ChunkingConfig(BaseConfig):
    preprocess: bool = False
    chunk_size: int = 512
    overlap_size: int = 128
    generate_title: bool = False
    generate_summary: bool = False
    override_content_with_summary: bool = False
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.BASIC
    # only for azure document intelligence strategy
    azure_document_intelligence_model: str = "prebuilt-read"
