from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class RerankerConfig(BaseConfig):
    type: str = "crossencoder"
    cross_encoder_at_k: int = 3
    crossencoder_model: str = ""
    llm_rerank_threshold: int = 3
