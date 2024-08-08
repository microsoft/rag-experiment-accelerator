from enum import StrEnum
import json
import os

from dataclasses import dataclass, field

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.path_config import PathConfig
from rag_experiment_accelerator.config.sampling_config import SamplingConfig
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.base_config import BaseConfig
from rag_experiment_accelerator.config.language_config import LanguageConfig
from rag_experiment_accelerator.config.reranker_config import RerankerConfig
from rag_experiment_accelerator.config.search_config import SearchConfig
from rag_experiment_accelerator.config.query_expansion import QueryExpansionConfig
from rag_experiment_accelerator.config.openai_config import OpenAIConfig
from rag_experiment_accelerator.config.eval_config import EvalConfig

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.embedding.factory import create_embedding_model
from rag_experiment_accelerator.utils.logging import get_logger

from rag_experiment_accelerator.llm.prompt import main_instruction


logger = get_logger(__name__)


class ExecutionEnvironment(StrEnum):
    LOCAL = "local"
    AZURE_ML = "azure-ml"


@dataclass
class Config(BaseConfig):
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL
    experiment_name: str = ""
    job_name: str = ""
    job_description: str = ""
    data_formats: list[str] = field(default_factory=lambda: ["*"])
    main_instruction: str = ""
    max_worker_threads: int = None
    use_checkpoints: bool = True
    path: PathConfig = field(default_factory=PathConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    rerank: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_path(
        cls, environment: Environment, config_path: str = None, data_dir: str = None
    ) -> "Config":
        if not config_path:
            config_path = os.path.join(os.getcwd(), "./config.json")
        with open(config_path.strip(), "r") as json_file:
            config_json: dict[str, any] = json.load(json_file)

        # check type casting
        config = Config.from_dict(config_json)

        config.path.initialize_paths(config_path, data_dir)

        if not config.main_instruction:
            config.main_instruction = main_instruction

        config.validate_inputs(
            use_semantic_search=environment.azure_search_use_semantic_search.lower()
            == "true"
        )

        config.initialize_embedding_models(environment)

        config.execution_environment = ExecutionEnvironment.LOCAL
        max_worker_threads = os.environ.get("MAX_WORKER_THREADS", None)
        if max_worker_threads:
            config.max_worker_threads = int(max_worker_threads)

        # log all the configuration settings in debug mode
        for key, value in config.to_dict():
            logger.debug(f"Configuration setting: {key} = {value}")

        return config

    def validate_inputs(self, use_semantic_search: bool):
        if any(val < 100 or val > 1000 for val in self.index.ef_construction):
            raise ValueError(
                "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
            )
        if any(val < 100 or val > 1000 for val in self.index.ef_search):
            raise ValueError(
                "Config param validation error: ef_search must be between 100 and 1000 (inclusive)"
            )
        if max(self.index.chunking.overlap_size) > min(self.index.chunking.chunk_size):
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

    def initialize_embedding_models(self, environment: Environment):
        self.__embedding_models_dictionary = {}
        for model_config in self.index.embedding_model:
            kwargs = {"environment": environment, **model_config.to_dict()}
            self.__embedding_models_dictionary[
                model_config.model_name
            ] = create_embedding_model(model_config.type, **kwargs)

    def get_embedding_model(self, model_name) -> EmbeddingModel:
        return self.__embedding_models_dictionary.get(model_name)
