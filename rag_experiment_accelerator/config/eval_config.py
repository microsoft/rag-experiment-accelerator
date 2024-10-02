from dataclasses import dataclass, field
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class EvalConfig(BaseConfig):
    metric_types: list[str] = field(
        default_factory=lambda: [
            "fuzzy_score",
            "bert_all_MiniLM_L6_v2",
            "cosine_ochiai",
            "bert_distilbert_base_nli_stsb_mean_tokens",
            "ragas_answer_relevance",
            "ragas_context_precision",
        ]
    )
