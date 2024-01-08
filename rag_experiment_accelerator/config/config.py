import json
import os

from rag_experiment_accelerator.config.credentials import (
    AzureMLCredentials,
    AzureSearchCredentials,
    AzureSkillsCredentials,
    OpenAICredentials,
)
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.embedding.factory import EmbeddingModelFactory
from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class Config:
    """
    A singleton class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'config.json'.

    Attributes:
        CHUNK_SIZES (list[int]): A list of integers representing the chunk sizes for chunking documents.
        OVERLAP_SIZES (list[int]): A list of integers representing the overlap sizes for chunking documents.
        EMBEDDING_DIMENSIONS (list[int]): The number of dimensions to use for document embeddings.
        EF_CONSTRUCTIONS (list[int]): The number of ef_construction to use for HNSW index.
        EF_SEARCHES (list[int]): The number of ef_search to use for HNSW index.
        NAME_PREFIX (str): A prefix to use for the names of saved models.
        SEARCH_VARIANTS (list[str]): A list of search types to use.
        AZURE_OAI_CHAT_DEPLOYMENT_NAME (str): The name of the Azure deployment to use.
        EMBEDDING_MODEL_NAME (str): The name of the Azure deployment to use for embeddings.
        AZURE_OAI_EVAL_DEPLOYMENT_NAME (str): The name of the deployment to use for evaluation.
        RETRIEVE_NUM_OF_DOCUMENTS (int): The number of documents to retrieve for each query.
        CROSSENCODER_MODEL (str): The name of the crossencoder model to use.
        RERANK_TYPE (str): The type of reranking to use.
        LLM_RERANK_THRESHOLD (float): The threshold for reranking using LLM.
        CROSSENCODER_AT_K (int): The number of documents to rerank using the crossencoder.
        TEMPERATURE (float): The temperature to use for OpenAI's GPT-3 model.
        RERANK (bool): Whether or not to perform reranking.
        SEARCH_RELEVANCY_THRESHOLD (float): The threshold for search result relevancy.
        DATA_FORMATS (Union[list[str], str]): Allowed formats for input data, if "all", then all formats will be loaded"
        METRIC_TYPES (list[str]): A list of metric types to use.
        EVAL_DATA_JSONL_FILE_PATH (str): File path for eval data jsonl file which is input for 03_querying script
    """

    _instance = None

    def __new__(cls, config_dir: str = os.getcwd()):
        """
        Creates a new instance of Config only if it doesn't already exist.

        Parameters:
            config_filename (str): The name of the JSON file containing configuration settings.

        Returns:
            Config: The singleton instance of Config.
        """

        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize(config_dir)
        return cls._instance

    def _initialize(self, config_dir: str) -> None:
        with open(f"{config_dir}/config.json", "r") as json_file:
            data = json.load(json_file)

        self.config_dir = config_dir
        self.artifacts_dir = f"{config_dir}/artifacts"
        self.data_dir = f"{config_dir}/data"
        self.EVAL_DATA_JSONL_FILE_PATH = (
            f"{self.config_dir}/{data['eval_data_jsonl_file_path']}"
        )
        self.CHUNK_SIZES = data["chunking"]["chunk_size"]
        self.OVERLAP_SIZES = data["chunking"]["overlap_size"]
        self.EF_CONSTRUCTIONS = data["ef_construction"]
        self.EF_SEARCHES = data["ef_search"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.AZURE_OAI_CHAT_DEPLOYMENT_NAME = data.get(
            "azure_oai_chat_deployment_name", None
        )
        self.EMBEDDING_MODEL_NAME = data.get("embedding_model_name", None)
        self.AZURE_OAI_EVAL_DEPLOYMENT_NAME = data.get(
            "azure_oai_eval_deployment_name", None
        )
        self.RETRIEVE_NUM_OF_DOCUMENTS = data["retrieve_num_of_documents"]
        self.CROSSENCODER_MODEL = data["crossencoder_model"]
        self.RERANK_TYPE = data["rerank_type"]
        self.LLM_RERANK_THRESHOLD = data["llm_re_rank_threshold"]
        self.CROSSENCODER_AT_K = data["cross_encoder_at_k"]
        self.TEMPERATURE = data["openai_temperature"]
        self.RERANK = data["rerank"]
        self.SEARCH_RELEVANCY_THRESHOLD = data.get("search_relevancy_threshold", 0.8)
        self.DATA_FORMATS = data.get("data_formats", "all")
        self.METRIC_TYPES = data["metric_types"]
        self.LANGUAGE = data.get("language", {})
        self.OpenAICredentials = OpenAICredentials.from_env()
        self.AzureSearchCredentials = AzureSearchCredentials.from_env()
        self.AzureMLCredentials = AzureMLCredentials.from_env()
        self.AzureSkillsCredentials = AzureSkillsCredentials.from_env()

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = data.get("embedding_models", [])
        for model_config in embedding_model_config:
            kwargs = {"openai_creds": self.OpenAICredentials, **model_config}
            self.embedding_models.append(EmbeddingModelFactory.create(**kwargs))

        try:
            with open(f"{config_dir}/prompt_config.json", "r") as json_file:
                data = json.load(json_file)

            self.MAIN_PROMPT_INSTRUCTION = data["main_prompt_instruction"]
            if self.MAIN_PROMPT_INSTRUCTION is None:
                logger.warn(
                    "prompt_config.json found but main_prompt_instruction is"
                    " not set. Using default prompts"
                )
                self.MAIN_PROMPT_INSTRUCTION = main_prompt_instruction
        except OSError:
            logger.warn("prompt_config.json not found. Using default prompts")
            self.MAIN_PROMPT_INSTRUCTION = main_prompt_instruction
