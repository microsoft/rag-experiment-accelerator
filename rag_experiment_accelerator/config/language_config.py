from dataclasses import dataclass, field
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class LanguageAnalyzerConfig(BaseConfig):
    analyzer_name: str = "en.microsoft"
    index_analyzer_name: str = ""
    search_analyzer_name: str = ""
    char_filters: list[any] = field(default_factory=list)
    tokenizers: list[any] = field(default_factory=list)
    token_filters: list[any] = field(default_factory=list)

    def __post_init__(self):
        super().__init__()


@dataclass
class LanguageConfig(BaseConfig):
    analyzers: LanguageAnalyzerConfig = field(default_factory=LanguageAnalyzerConfig)
    query_language: str = "en-us"

    def __post_init__(self):
        super().__init__()
