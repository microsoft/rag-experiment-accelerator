from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class RerankConfig(BaseConfig):
    enabled: bool = False  # todo: consider moving it as a module switch
    type: str = "cross_encoder"
    cross_encoder_at_k: int = 3
    cross_encoder_model: str = ""
    llm_rerank_threshold: int = 3
