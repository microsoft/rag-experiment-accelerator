from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class SearchConfig(BaseConfig):
    retrieve_num_of_documents: int = 3
    search_type: str = "search_for_match_semantic"
    search_relevancy_threshold: float = 0.8
