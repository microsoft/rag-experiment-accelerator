from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class EmbeddingModelConfig(BaseConfig):
    type: str = "sentence-transformer"
    model_name: str = "all-mpnet-base-v2"

    def __post_init__(self):
        super().__init__()
