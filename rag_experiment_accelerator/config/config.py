import json
import os

from dataclasses import dataclass, field

from rag_experiment_accelerator.config.path_config import PathConfig
from rag_experiment_accelerator.config.sampling_config import SamplingConfig
from rag_experiment_accelerator.config.environment import Environment

from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.base_config import BaseConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)
from rag_experiment_accelerator.config.language_config import LanguageConfig
from rag_experiment_accelerator.config.reranker_config import RerankerConfig
from rag_experiment_accelerator.config.search_config import SearchConfig
from rag_experiment_accelerator.config.query_expansion import QueryExpansionConfig
from rag_experiment_accelerator.config.openai_config import OpenAIConfig
from rag_experiment_accelerator.config.eval_config import EvalConfig

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class Config(BaseConfig):
    experiment_name: str = ""
    job_name: str = ""
    job_description: str = ""
    data_formats: list[str] = field(default_factory=lambda: ["*"])
    main_prompt_instruction: str = ""
    path: PathConfig = field(default_factory=PathConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    index_config: IndexConfig = field(default_factory=IndexConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    rerank: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        super().__init__()

    @classmethod
    def from_path(
        cls, environment: Environment, config_path: str = None, data_dir: str = None
    ) -> BaseConfig:
        if not config_path:
            config_path = os.path.join(os.getcwd(), "./config.json")
        with open(config_path.strip(), "r") as json_file:
            config_json = json.load(json_file)

        config = Config.from_dict(config_json)

        config.path.initialize_paths(config_path, data_dir)

        if not config.main_prompt_instruction:
            config.main_prompt_instruction = main_prompt_instruction

        config.validate_inputs(
            use_semantic_search=environment.azure_search_use_semantic_search.lower()
            == "true"
        )

        # config.embedding_models: list[EmbeddingModel] = []
        # embedding_model_config = config_json.get("embedding_models", [])
        # for model_config in embedding_model_config:
        #    kwargs = {"environment": environment, **model_config}
        #    self.embedding_models.append(
        #        create_embedding_model(model_config["type"], **kwargs)
        #    )

        # max_worker_threads = os.environ.get("MAX_WORKER_THREADS", None)
        # self.MAX_WORKER_THREADS = (
        #     int(max_worker_threads) if max_worker_threads else None
        # )

        # log all the configuration settings in debug mode
        for key, value in config_json.items():
            logger.debug(f"Configuration setting: {key} = {value}")

        return config

    def validate_inputs(self, use_semantic_search: bool):
        if any(val < 100 or val > 1000 for val in self.index_config.ef_construction):
            raise ValueError(
                "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
            )
        if any(val < 100 or val > 1000 for val in self.index_config.ef_search):
            raise ValueError(
                "Config param validation error: ef_search must be between 100 and 1000 (inclusive)"
            )
        if max(self.index_config.chunking_config.overlap) > min(
            self.index_config.chunking_config.chunk_size
        ):
            raise ValueError(
                "Config param validation error: overlap_size must be less than chunk_size"
            )

        if self.sampling.sample_data and (
            self.sampling.sample_percentage < 0 or self.sampling.sample_percentage > 100
        ):
            raise ValueError(
                "Config param validation error: sample_percentage must be between 0 and 100 (inclusive)"
            )

        if (
            "search_for_match_semantic" in self.search.search_type
            or "search_for_manual_hybrid" in self.search.search_type
        ) and not use_semantic_search:
            raise ValueError(
                "Semantic search is required for search types 'search_for_match_semantic' or 'search_for_manual_hybrid', but it's not enabled."
            )

    def _find_embedding_model_by_name(self, model_name: str) -> EmbeddingModel:
        for model in self.embedding_model:
            if model.name == model_name:
                return model
        raise AttributeError(f"No model found with the name: [{model_name}]")

    def _sampled_cluster_predictions_path(self):
        return os.path.join(
            self.path.sampling_output_dir,
            f"sampled_cluster_predictions_cluster_number_{self.sampling.optimum_k}.csv",
        )
