import json
import os
from typing import Generator
from enum import StrEnum

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.embedding.factory import create_embedding_model
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

from rag_experiment_accelerator.llm.prompt import main_instruction


logger = get_logger(__name__)


class ChunkingStrategy(StrEnum):
    BASIC = "basic"
    AZURE_DOCUMENT_INTELLIGENCE = "azure-document-intelligence"

    def __repr__(self) -> str:
        return f'"{self.value}"'


class ExecutionEnvironment(StrEnum):
    LOCAL = "local"
    AZURE_ML = "azure-ml"


class Config:
    """
    A class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'config.json'.
        data_dir (str):
            An input path to read data from. Can be local directory, or AzureML-supported URL when running on AzureML.
            Defaults to "./data".

    Attributes:
        preprocess (bool): Whether or not to preprocess the text (converting it to lowercase, removing punctuation and tags, removing stop words, and tokenizing the words).
        chunk_sizes (list[int]): A list of integers representing the chunk sizes for chunking documents.
        overlap_sizes (list[int]): A list of integers representing the overlap sizes for chunking documents.
        generate_title (bool): Whether or not to generate title for chunk content. Default is False.
        generate_summary (bool): Whether or not to generate summary for chunk content. Default is False.
        override_content_with_summary (bool): Whether or not to override chunk content with generated summary. Default is False.
        ef_constructions (list[int]): The number of ef_construction to use for HNSW index.
        ef_searches (list[int]): The number of ef_search to use for HNSW index.
        index_name_prefix (str): A prefix to use for the names of saved models.
        experiment_name (str): The name of the experiment in Azure ML (optional, if not set index_name_prefix will be used).
        job_name (str): The name of the job in Azure ML (optional, if not set experiment_name and current datetime will be used).
        job_description (str): The description of the job in Azure ML (optional).
        search_types (list[str]): A list of search types to use.
        azure_oai_chat_deployment_name (str): The name of the Azure deployment to use.
        azure_oai_eval_deployment_name (str): The name of the deployment to use for evaluation.
        retrieve_num_of_documents (int): The number of documents to retrieve for each query.
        crossencoder_model (str): The name of the crossencoder model to use.
        rerank_type (str): The type of reranking to use.
        llm_rerank_threshold (float): The threshold for reranking using LLM.
        crossencoder_at_k (int): The number of documents to rerank using the crossencoder.
        chunking_strategy (ChunkingStrategy): The strategy to use for chunking documents.
        azure_document_intelligence_model (str): The model to use for Azure Document Intelligence extraction.
        temperature (float): The temperature to use for OpenAI's GPT-3 model.
        rerank (bool): Whether or not to perform reranking.
        search_relevancy_threshold (float): The threshold for search result relevancy.
        data_formats (Union[list[str], str]): Allowed formats for input data, if "all", then all formats will be loaded.
        metric_types (list[str]): A list of metric types to use.
        eval_data_jsonl_file_path (str): File path for eval data jsonl file which is input for 03_querying script.
        embedding_models: The embedding models used to generate embeddings.
        max_worker_threads (int): Maximum number of worker threads.
        sampling (bool): Sample the dataset in accordance to the content and structure distribution.
        sample_percentage (int): Percentage of dataset.
        expand_to_multiple_questions (bool): Whether expanding to multiple questions is enabled or not. if enabled LLM will check if it's possible to split complex query to multiple queries and do so. else it will leave the original query as is. Default is False.
        hyde (str): Whether or not to generate hypothetical answer or document which holds an answer for the query using LLM. Possible values are "disabled", "generated_hypothetical_answer", "generated_hypothetical_document_to_answer". Default is 'disabled'.
        query_expansion (bool): Whether or not to perform query expansion and generate up to five related questions using LLM (depends on similairy score) and use those to retrieve documents. Default is False.
        min_query_expansion_related_question_similarity_score (int): The minimum similarity score for query expansion generated related questions. Default is 90.
    """

    def __init__(
        self, environment: Environment, config_path: str = None, data_dir: str = None
    ):
        if not config_path:
            config_path = os.path.join(os.getcwd(), "./config.json")
        if not data_dir:
            data_dir = os.path.join(os.getcwd(), "data/")
        with open(config_path.strip(), "r") as json_file:
            config_json: dict[str, any] = json.load(json_file)

        self._initialize_paths(config_json, config_path, data_dir)
        self.preprocess = config_json.get("preprocess", False)
        chunking_config = config_json["chunking"]
        self.chunk_sizes = chunking_config["chunk_size"]
        self.overlap_sizes = chunking_config["overlap_size"]
        self.generate_title = chunking_config.get("generate_title", False)
        self.generate_summary = chunking_config.get("generate_summary", False)
        self.override_content_with_summary = chunking_config.get(
            "override_content_with_summary", False
        )
        self.ef_constructions = config_json["ef_construction"]
        self.ef_searches = config_json["ef_search"]
        self.index_name_prefix = config_json["index_name_prefix"]
        self.experiment_name = (
            config_json.get("experiment_name", "") or self.index_name_prefix
        )
        self.job_name = config_json["job_name"]
        self.job_description = config_json["job_description"]
        self.search_types = config_json["search_types"]
        self.azure_oai_chat_deployment_name = config_json.get(
            "azure_oai_chat_deployment_name", None
        )
        self.azure_oai_eval_deployment_name = config_json.get(
            "azure_oai_eval_deployment_name", None
        )
        self.use_checkpoints = config_json.get("use_checkpoints", True)
        self.retrieve_num_of_documents = config_json["retrieve_num_of_documents"]
        self.crossencoder_model = config_json["crossencoder_model"]
        self.rerank_type = config_json["rerank_type"]
        self.llm_rerank_threshold = config_json["llm_re_rank_threshold"]
        self.crossencoder_at_k = config_json["cross_encoder_at_k"]
        self.temperature = config_json["openai_temperature"]
        self.rerank = config_json["rerank"]
        self.search_relevency_threshold = config_json.get(
            "search_relevancy_threshold", 0.8
        )
        self.data_formats = config_json.get("data_formats", "all")
        self.metric_types = config_json["metric_types"]
        self.chunking_strategy = (
            ChunkingStrategy(config_json["chunking_strategy"])
            if "chunking_strategy" in config_json
            else ChunkingStrategy.BASIC
        )
        self.azure_document_intelligence_model = config_json.get(
            "azure_document_intelligence_model", "prebuilt-read"
        )
        self.execution_environment = ExecutionEnvironment.LOCAL
        self.language = config_json.get("language", {})

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = config_json.get("embedding_models", [])
        for model_config in embedding_model_config:
            kwargs = {"environment": environment, **model_config}
            self.embedding_models.append(
                create_embedding_model(model_config["type"], **kwargs)
            )

        max_worker_threads = os.environ.get("MAX_WORKER_THREADS", None)
        self.max_worker_threads = (
            int(max_worker_threads) if max_worker_threads else None
        )

        self.validate_inputs(
            self.chunk_sizes,
            self.overlap_sizes,
            self.ef_constructions,
            self.ef_searches,
        )

        if "main_prompt_instruction" in config_json:
            main_instruction.update_system_prompt(
                config_json["main_prompt_instruction"]
            )

        self.SAMPLE_DATA = "sampling" in config_json
        if self.sampling:
            self.sample_percentage = config_json["sampling"]["sample_percentage"]
            if self.sample_percentage < 0 or self.sample_percentage > 100:
                raise ValueError(
                    "Config param validation error: sample_percentage must be between 0 and 100 (inclusive)"
                )
            self.sample_optimum_k = config_json["sampling"]["optimum_k"]
            self.sample_min_cluster = config_json["sampling"]["min_cluster"]
            self.sample_max_cluster = config_json["sampling"]["max_cluster"]

        self.hyde = config_json.get("hyde", "disabled").lower()
        self.query_expansion = config_json.get("query_expansion", False)
        self.min_query_expansion_related_question_similarity_score = int(
            config_json.get("min_query_expansion_related_question_similarity_score", 90)
        )
        self.expand_to_multiple_questions = config_json.get(
            "expand_to_multiple_questions", False
        )
        self.validate_semantic_search_config(
            environment.azure_search_use_semantic_search.lower() == "true"
        )

        # log all the configuration settings in debug mode
        for key, value in config_json.items():
            logger.debug(f"Configuration setting: {key} = {value}")

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
            "search_for_match_semantic" in self.search_types
            or "search_for_manual_hybrid" in self.search_types
        ) and not use_semantic_search:
            raise ValueError(
                "Semantic search is required for search variants 'search_for_match_semantic' or 'search_for_manual_hybrid', but it's not enabled."
            )

    def index_configs(self) -> Generator[IndexConfig, None, None]:
        for chunk_size in self.chunk_sizes:
            for overlap in self.overlap_sizes:
                for embedding_model in self.embedding_models:
                    for ef_construction in self.ef_constructions:
                        for ef_search in self.ef_searches:
                            yield IndexConfig(
                                index_name_prefix=self.index_name_prefix,
                                preprocess=self.preprocess,
                                chunk_size=chunk_size,
                                overlap_size=overlap,
                                embedding_model=embedding_model,
                                ef_construction=ef_construction,
                                ef_search=ef_search,
                                sampling_percentage=self.sample_percentage
                                if self.sampling
                                else 0,
                                generate_title=self.generate_title,
                                generate_summary=self.generate_summary,
                                override_content_with_summary=self.override_content_with_summary,
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

        self.eval_data_jsonl_file_path = (
            config_json["eval_data_jsonl_file_path"]
            if "eval_data_jsonl_file_path" in config_json
            else os.path.join(self.artifacts_dir, "eval_data.jsonl")
        )
        self.generated_index_names_file_path = (
            config_json["generated_index_names_file_path"]
            if "generated_index_names_file_path" in config_json
            else os.path.join(self.artifacts_dir, "generated_index_names.jsonl")
        )
        self.query_data_location = (
            config_json["query_data"]
            if "query_data" in config_json
            else os.path.join(self.artifacts_dir, "query_data")
        )
        self._try_create_directory(self.query_data_location)

        self.eval_data_location = (
            config_json["eval_data"]
            if "eval_data" in config_json
            else os.path.join(self.artifacts_dir, "eval_score")
        )
        self._try_create_directory(self.eval_data_location)

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
            f"sampled_cluster_predictions_cluster_number_{self.sample_optimum_k}.csv",
        )
