import json
import os
from typing import Generator
from enum import StrEnum

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.embedding.factory import create_embedding_model
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment


logger = get_logger(__name__)


class ChunkingStrategy(StrEnum):
    BASIC = "basic"
    AZURE_DOCUMENT_INTELLIGENCE = "azure-document-intelligence"


class Config:
    """
    A class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'config.json'.
        data_dir (str):
            An input path to read data from. Can be local directory, or AzureML-supported URL when running on AzureML.
            Defaults to "./data".

    Attributes:
        PREPROCESS (bool): Whether or not to preprocess the text (converting it to lowercase, removing punctuation and tags, removing stop words, and tokenizing the words).
        CHUNK_SIZES (list[int]): A list of integers representing the chunk sizes for chunking documents.
        OVERLAP_SIZES (list[int]): A list of integers representing the overlap sizes for chunking documents.
        GENERATE_TITLE (bool): Whether or not to generate title for chunk content. Default is False.
        GENERATE_SUMMARY (bool): Whether or not to generate summary for chunk content. Default is False.
        OVERRIDE_CONTENT_WITH_SUMMARY (bool): Whether or not to override chunk content with generated summary. Default is False.
        EF_CONSTRUCTIONS (list[int]): The number of ef_construction to use for HNSW index.
        EF_SEARCHES (list[int]): The number of ef_search to use for HNSW index.
        INDEX_NAME_PREFIX (str): A prefix to use for the names of saved models.
        EXPERIMENT_NAME (str): The name of the experiment in Azure ML (optional, if not set INDEX_NAME_PREFIX will be used).
        JOB_NAME (str): The name of the job in Azure ML (optional, if not set EXPERIMENT_NAME and current datetime will be used).
        JOB_DESCRIPTION (str): The description of the job in Azure ML (optional).
        SEARCH_VARIANTS (list[str]): A list of search types to use.
        AZURE_OAI_CHAT_DEPLOYMENT_NAME (str): The name of the Azure deployment to use.
        AZURE_OAI_EVAL_DEPLOYMENT_NAME (str): The name of the deployment to use for evaluation.
        RETRIEVE_NUM_OF_DOCUMENTS (int): The number of documents to retrieve for each query.
        CROSSENCODER_MODEL (str): The name of the crossencoder model to use.
        RERANK_TYPE (str): The type of reranking to use.
        LLM_RERANK_THRESHOLD (float): The threshold for reranking using LLM.
        CROSSENCODER_AT_K (int): The number of documents to rerank using the crossencoder.
        CHUNKING_STRATEGY (ChunkingStrategy): The strategy to use for chunking documents.
        AZURE_DOCUMENT_INTELLIGENCE_MODEL (str): The model to use for Azure Document Intelligence extraction.
        TEMPERATURE (float): The temperature to use for OpenAI's GPT-3 model.
        RERANK (bool): Whether or not to perform reranking.
        SEARCH_RELEVANCY_THRESHOLD (float): The threshold for search result relevancy.
        DATA_FORMATS (Union[list[str], str]): Allowed formats for input data, if "all", then all formats will be loaded.
        METRIC_TYPES (list[str]): A list of metric types to use.
        EVAL_DATA_JSONL_FILE_PATH (str): File path for eval data jsonl file which is input for 03_querying script.
        embedding_models: The embedding models used to generate embeddings.
        MAX_WORKER_THREADS (int): Maximum number of worker threads.
        SAMPLE_DATA (bool): Sample the dataset in accordance to the content and structure distribution.
        SAMPLE_PERCENTAGE (int): Percentage of dataset.
        CHAIN_OF_THOUGHTS (bool): Whether chain of thoughts is enabled or not. if enabled LLM will check if it's possible to split complex query to multiple queries and do so. else it will leave the original query as is. Default is False.
        HYDE (str): Whether or not to generate hypothetical answer or document which holds an answer for the query using LLM. Possible values are "disabled", "generated_hypothetical_answer", "generated_hypothetical_document_to_answer". Default is 'disabled'.
        QUERY_EXPANSION (bool): Whether or not to perform query expansion and generate up to five related questions using LLM (depends on similairy score) and use those to retrieve documents. Default is False.
        MIN_QUERY_EXPANSION_RELATED_QUESTION_SIMILARITY_SCORE (int): The minimum similarity score for query expansion generated related questions. Default is 90.
    """

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
        self.PREPROCESS = config_json.get("preprocess", False)
        chunking_config = config_json["chunking"]
        self.CHUNK_SIZES = chunking_config["chunk_size"]
        self.OVERLAP_SIZES = chunking_config["overlap_size"]
        self.GENERATE_TITLE = chunking_config.get("generate_title", False)
        self.GENERATE_SUMMARY = chunking_config.get("generate_summary", False)
        self.OVERRIDE_CONTENT_WITH_SUMMARY = chunking_config.get(
            "override_content_with_summary", False
        )
        self.EF_CONSTRUCTIONS = config_json["ef_construction"]
        self.EF_SEARCHES = config_json["ef_search"]
        self.INDEX_NAME_PREFIX = config_json["index_name_prefix"]
        self.EXPERIMENT_NAME = config_json["experiment_name"] or self.INDEX_NAME_PREFIX
        self.JOB_NAME = config_json["job_name"]
        self.JOB_DESCRIPTION = config_json["job_description"]
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
        self.CHUNKING_STRATEGY = (
            ChunkingStrategy(config_json["chunking_strategy"])
            if "chunking_strategy" in config_json
            else ChunkingStrategy.BASIC
        )
        self.AZURE_DOCUMENT_INTELLIGENCE_MODEL = config_json.get(
            "azure_document_intelligence_model", "prebuilt-read"
        )
        self.LANGUAGE = config_json.get("language", {})

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = config_json.get("embedding_models", [])
        for model_config in embedding_model_config:
            kwargs = {"environment": environment, **model_config}
            self.embedding_models.append(
                create_embedding_model(model_config["type"], **kwargs)
            )

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

    def index_configs(self) -> Generator[IndexConfig, None, None]:
        for chunk_size in self.CHUNK_SIZES:
            for overlap in self.OVERLAP_SIZES:
                for embedding_model in self.embedding_models:
                    for ef_construction in self.EF_CONSTRUCTIONS:
                        for ef_search in self.EF_SEARCHES:
                            yield IndexConfig(
                                index_name_prefix=self.INDEX_NAME_PREFIX,
                                preprocess=self.PREPROCESS,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                embedding_model=embedding_model,
                                ef_construction=ef_construction,
                                ef_search=ef_search,
                                sampling_percentage=self.SAMPLE_PERCENTAGE
                                if self.SAMPLE_DATA
                                else 0,
                                generate_title=self.GENERATE_TITLE,
                                generate_summary=self.GENERATE_SUMMARY,
                                override_content_with_summary=self.OVERRIDE_CONTENT_WITH_SUMMARY,
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
