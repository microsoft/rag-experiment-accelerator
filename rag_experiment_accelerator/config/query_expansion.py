from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class QueryExpansionConfig(BaseConfig):
    # todo: refactor the settings to be more descriptive
    query_expansion: bool = False
    expand_to_multiple_questions: bool = False
    min_query_expansion_related_question_similarity_score: int = 90
    hyde: str = "disabled"
