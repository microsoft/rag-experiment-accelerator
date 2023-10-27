import json
import os
from utils.logging import get_logger

logger = get_logger(__name__)


def _mask_string(s: str, start: int = 2, end: int = 2, mask_char: str = "*") -> str:
    if s == "":
        return ""

    if len(s) <= start + end:
        return s[0] + "*" * (len(s) - 1)

    return s[:start] + mask_char * (len(s) - start - end) + s[-end:]


def _get_env_var(var_name: str, critical: bool, mask: bool) -> str:
    var = os.getenv(var_name, None)
    if var is None:
        logger.critical(f"{var_name} environment variable not set.")
        if critical:
            raise ValueError(f"{var_name} environment variable not set.")
    else:
        text = var if not mask else _mask_string(var)
        logger.info(f"{var_name} set to {text}")
    return var


class AzureSearchCredentials:
    def __init__(
        self,
        azure_search_service_endpoint: str,
        azure_search_admin_key: str,
    ) -> None:
        self.AZURE_SEARCH_SERVICE_ENDPOINT = azure_search_service_endpoint
        self.AZURE_SEARCH_ADMIN_KEY = azure_search_admin_key

    @classmethod
    def from_env(cls) -> "AzureSearchCredentials":
        return cls(
            azure_search_service_endpoint=_get_env_var(
                var_name="AZURE_SEARCH_SERVICE_ENDPOINT",
                critical=False,
                mask=False,
            ),
            azure_search_admin_key=_get_env_var(
                var_name="AZURE_SEARCH_ADMIN_KEY",
                critical=False,
                mask=True,
            ),
        )


class AzureMLCredentials:
    def __init__(
        self,
        subscription_id: str,
        workspace_name: str,
        resource_group_name: str,
    ) -> None:
        self.SUBSCRIPTION_ID = subscription_id
        self.WORKSPACE_NAME = workspace_name
        self.RESOURCE_GROUP_NAME = resource_group_name

    @classmethod
    def from_env(cls) -> "AzureMLCredentials":
        return cls(
            subscription_id=_get_env_var(
                var_name="SUBSCRIPTION_ID",
                critical=False,
                mask=True,
            ),
            workspace_name=_get_env_var(
                var_name="WORKSPACE_NAME",
                critical=False,
                mask=False,
            ),
            resource_group_name=_get_env_var(
                var_name="RESOURCE_GROUP_NAME",
                critical=False,
                mask=False,
            ),
        )


class OpenAICredentials:
    def __init__(
        self,
        openai_api_type: str,
        openai_api_key: str,
        openai_api_version: str,
        openai_endpoint: str,
    ) -> None:
        if openai_api_type is not None and openai_api_type not in ["azure", "openai"]:
            logger.critical("OPENAI_API_TYPE must be either 'azure' or 'openai'.")
            raise ValueError("OPENAI_API_TYPE must be either 'azure' or 'openai'.")

        self.OPENAI_API_TYPE = openai_api_type
        self.OPENAI_API_KEY = openai_api_key
        self.OPENAI_API_VERSION = openai_api_version
        self.OPENAI_ENDPOINT = openai_endpoint

        self._set_credentials()

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        return cls(
            openai_api_type=_get_env_var(
                var_name="OPENAI_API_TYPE",
                critical=False,
                mask=False,
            ),
            openai_api_key=_get_env_var(
                var_name="OPENAI_API_KEY", critical=False, mask=True
            ),
            openai_api_version=_get_env_var(
                var_name="OPENAI_API_VERSION",
                critical=False,
                mask=False,
            ),
            openai_endpoint=_get_env_var(
                var_name="OPENAI_ENDPOINT",
                critical=False,
                mask=True,
            ),
        )

    def _set_credentials(self) -> None:
        if self.OPENAI_API_TYPE is not None:
            import openai

            if self.OPENAI_API_TYPE == "azure":
                openai.api_type = "azure"
                openai.api_version = self.OPENAI_API_VERSION
                openai.api_base = self.OPENAI_ENDPOINT
            if self.OPENAI_API_TYPE is not None:
                openai.api_key = self.OPENAI_API_KEY


class Config:
    """
    A class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'search_config.json'.

    Attributes:
        CHUNK_SIZES (list[int]): A list of integers representing the chunk sizes for chunking documents.
        OVERLAP_SIZES (list[int]): A list of integers representing the overlap sizes for chunking documents.
        EMBEDDING_DIMENSIONS (list[int]): The number of dimensions to use for document embeddings.
        EF_CONSTRUCTIONS (list[int]): The number of efConstructions to use for HNSW index.
        EF_SEARCHES (list[int]): The number of efSearch to use for HNSW index.
        NAME_PREFIX (str): A prefix to use for the names of saved models.
        SEARCH_VARIANTS (list[str]): A list of search types to use.
        CHAT_MODEL_NAME (str): The name of the chat model to use.
        EMBEDDING_MODEL_NAME (str): The name of the Azure deployment to use for embeddings.
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
    """

    def __init__(self, config_filename: str = "search_config.json") -> None:
        with open(config_filename, "r") as json_file:
            data = json.load(json_file)

        self.CHUNK_SIZES = data["chunking"]["chunk_size"]
        self.OVERLAP_SIZES = data["chunking"]["overlap_size"]
        self.EMBEDDING_DIMENSIONS = data["embedding_dimension"]
        self.EF_CONSTRUCTIONS = data["efConstruction"]
        self.EF_SEARCHES = data["efSearch"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.CHAT_MODEL_NAME = data.get("chat_model_name", None)
        self.EMBEDDING_MODEL_NAME = data.get("embedding_model_name", None)
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
        self.OpenAICredentials = OpenAICredentials.from_env()
        self.AzureSearchCredentials = AzureSearchCredentials.from_env()
        self.AzureMLCredentials = AzureMLCredentials.from_env()
