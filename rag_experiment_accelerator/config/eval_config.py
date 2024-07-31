from dataclasses import dataclass, field
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class EvalConfig(BaseConfig):
    metric_types: list[str] = field(
        default_factory=lambda: [
            "fuzzy",
            "bert_all_MiniLM_L6_v2",
            "cosine",
            "bert_distilbert_base_nli_stsb_mean_tokens",
            "llm_answer_relevance",
            "llm_context_precision",
        ]
    )
    eval_data_jsonl_file_path: str = "./artifacts/eval_data.jsonl"

    def __post_init__(self):
        super().__init__()
