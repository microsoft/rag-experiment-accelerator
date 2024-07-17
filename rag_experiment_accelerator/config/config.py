import json
import os

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.utils.logging import get_logger
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


logger = get_logger(__name__)


class Config(BaseConfig):
    experiment_name: str
    job_name: str
    job_description: str
    data_formats: str | list[str]
    index_config: IndexConfig
    embedding_model: EmbeddingModelConfig
    language: LanguageConfig
    rerank: RerankerConfig
    search: SearchConfig
    query_expansion: QueryExpansionConfig
    openai: OpenAIConfig
    eval: EvalConfig

    def __init__(
        self, environment: Environment, config_path: str = None, data_dir: str = None
    ):
        if not config_path:
            config_path = os.path.join(os.getcwd(), "./config.json")
        if not data_dir:
            data_dir = os.path.join(os.getcwd(), "data/")
        with open(config_path.strip(), "r") as json_file:
            config_json = json.load(json_file)

        self._initialize_paths(config_json, config_path, data_dir)

        self.SEARCH_VARIANTS = config_json["search_types"]
        self.AZURE_OAI_CHAT_DEPLOYMENT_NAME = config_json.get(
            "azure_oai_chat_deployment_name", None
        )
        self.AZURE_OAI_EVAL_DEPLOYMENT_NAME = config_json.get(
            "azure_oai_eval_deployment_name", None
        )
        self.RETRIEVE_NUM_OF_DOCUMENTS = config_json["retrieve_num_of_documents"]
        self.CROSSENCODER_MODEL = config_json["crossencoder_model"]
        self.RERANK_TYPE = config_json["rerank_type"]
        self.LLM_RERANK_THRESHOLD = config_json["llm_re_rank_threshold"]
        self.CROSSENCODER_AT_K = config_json["cross_encoder_at_k"]
        self.TEMPERATURE = config_json["openai_temperature"]
        self.RERANK = config_json["rerank"]
        self.SEARCH_RELEVANCY_THRESHOLD = config_json.get(
            "search_relevancy_threshold", 0.8
        )
        self.DATA_FORMATS = config_json.get("data_formats", "all")
        self.METRIC_TYPES = config_json["metric_types"]
        self.AZURE_DOCUMENT_INTELLIGENCE_MODEL = config_json.get(
            "azure_document_intelligence_model", "prebuilt-read"
        )
        self.LANGUAGE = config_json.get("language", {})

        # self.embedding_models: list[EmbeddingModel] = []
        # embedding_model_config = config_json.get("embedding_models", [])
        # for model_config in embedding_model_config:
        #    kwargs = {"environment": environment, **model_config}
        #    self.embedding_models.append(
        #        create_embedding_model(model_config["type"], **kwargs)
        #    )

        max_worker_threads = os.environ.get("MAX_WORKER_THREADS", None)
        self.MAX_WORKER_THREADS = (
            int(max_worker_threads) if max_worker_threads else None
        )

        self.validate_inputs(
            self.CHUNK_SIZES,
            self.OVERLAP_SIZES,
            self.EF_CONSTRUCTIONS,
            self.EF_SEARCHES,
        )
        self.MAIN_PROMPT_INSTRUCTION = (
            config_json["main_prompt_instruction"]
            if "main_prompt_instruction" in config_json
            else main_prompt_instruction
        )
        self.SAMPLE_DATA = "sampling" in config_json
        if self.SAMPLE_DATA:
            self.SAMPLE_PERCENTAGE = config_json["sampling"]["sample_percentage"]
            if self.SAMPLE_PERCENTAGE < 0 or self.SAMPLE_PERCENTAGE > 100:
                raise ValueError(
                    "Config param validation error: sample_percentage must be between 0 and 100 (inclusive)"
                )
            self.SAMPLE_OPTIMUM_K = config_json["sampling"]["optimum_k"]
            self.SAMPLE_MIN_CLUSTER = config_json["sampling"]["min_cluster"]
            self.SAMPLE_MAX_CLUSTER = config_json["sampling"]["max_cluster"]

        # log all the configuration settings in debug mode
        for key, value in config_json.items():
            logger.debug(f"Configuration setting: {key} = {value}")

        self.CHAIN_OF_THOUGHTS = config_json.get("chain_of_thoughts", False)
        self.HYDE = config_json.get("hyde", "disabled").lower()
        self.QUERY_EXPANSION = config_json.get("generate_title", False)
        self.MIN_QUERY_EXPANSION_RELATED_QUESTION_SIMILARITY_SCORE = int(
            config_json.get("min_query_expansion_related_question_similarity_score", 90)
        )
        self.EXPAND_TO_MULTIPLE_QUESTIONS = config_json.get(
            "expand_to_multiple_questions", False
        )
        self.validate_semantic_search_config(
            environment.azure_search_use_semantic_search.lower() == "true"
        )

    def validate_inputs(self, chunk_size, overlap_size, ef_constructions, ef_searches):
        if any(val < 100 or val > 1000 for val in ef_constructions):
            raise ValueError(
                "Config param validation error: ef_construction must be between 100 and 1000 (inclusive)"
            )
        if any(val < 100 or val > 1000 for val in ef_searches):
            raise ValueError(
                "Config param validation error: ef_search must be between 100 and 1000 (inclusive)"
            )
        if max(overlap_size) > min(chunk_size):
            raise ValueError(
                "Config param validation error: overlap_size must be less than chunk_size"
            )

    def validate_semantic_search_config(self, use_semantic_search: bool):
        if (
            "search_for_match_semantic" in self.SEARCH_VARIANTS
            or "search_for_manual_hybrid" in self.SEARCH_VARIANTS
        ) and not use_semantic_search:
            raise ValueError(
                "Semantic search is required for search variants 'search_for_match_semantic' or 'search_for_manual_hybrid', but it's not enabled."
            )

    def _initialize_paths(
        self, config_json: dict[str], config_file_path: str, data_dir: str
    ) -> None:
        self._config_dir = os.path.dirname(config_file_path)

        self.artifacts_dir = (
            config_json["artifacts_dir"]
            if "artifacts_dir" in config_json
            else os.path.join(self._config_dir, "artifacts")
        )
        self._try_create_directory(self.artifacts_dir)

        if data_dir:
            self.data_dir = data_dir
        elif "data_dir" in config_json:
            self.data_dir = config_json["data_dir"]
        else:
            self.data_dir = os.path.join(self._config_dir, "data")

        self.EVAL_DATA_JSONL_FILE_PATH = (
            config_json["eval_data_jsonl_file_path"]
            if "eval_data_jsonl_file_path" in config_json
            else os.path.join(self.artifacts_dir, "eval_data.jsonl")
        )
        self.GENERATED_INDEX_NAMES_FILE_PATH = (
            config_json["generated_index_names_file_path"]
            if "generated_index_names_file_path" in config_json
            else os.path.join(self.artifacts_dir, "generated_index_names.jsonl")
        )
        self.QUERY_DATA_LOCATION = (
            config_json["query_data"]
            if "query_data" in config_json
            else os.path.join(self.artifacts_dir, "query_data")
        )
        self._try_create_directory(self.QUERY_DATA_LOCATION)

        self.EVAL_DATA_LOCATION = (
            config_json["eval_data"]
            if "eval_data" in config_json
            else os.path.join(self.artifacts_dir, "eval_score")
        )
        self._try_create_directory(self.EVAL_DATA_LOCATION)

        self.sampling_output_dir = (
            config_json["sampling_output_dir"]
            if "sampling_output_dir" in config_json
            else os.path.join(self.artifacts_dir, "sampling")
        )
        self._try_create_directory(self.sampling_output_dir)

    def _find_embedding_model_by_name(self, model_name: str) -> EmbeddingModel:
        for model in self.embedding_models:
            if model.name == model_name:
                return model
        raise AttributeError(f"No model found with the name: [{model_name}]")

    def _try_create_directory(self, directory: str) -> None:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            if "Read-only file system" in e.strerror:
                pass
            logger.warn(f"Failed to create directory {directory}: {e.strerror}")

    def _sampled_cluster_predictions_path(self):
        return os.path.join(
            self.sampling_output_dir,
            f"sampled_cluster_predictions_cluster_number_{self.SAMPLE_OPTIMUM_K}.csv",
        )
