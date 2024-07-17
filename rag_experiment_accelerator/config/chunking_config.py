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
    chunking_strategy: str = "basic"
    generate_title: bool = False
    generate_summary: bool = False
    override_content_with_summary: bool = False
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.BASIC
    azure_document_intelligence_model: str = "prebuilt-read"
