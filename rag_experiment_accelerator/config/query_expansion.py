from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class QueryExpansionConfig(BaseConfig):
    hyde: str = "disabled"
    chain_of_thoughts: bool = False
    query_expansion: bool = False
    min_query_expansion_related_question_similarity_score: int = 90
    expand_to_multiple_questions: bool = False
