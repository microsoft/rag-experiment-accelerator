import json
import os

from openai import AzureOpenAI, NotFoundError, OpenAI

from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def _mask_string(
    s: str, start: int = 2, end: int = 2, mask_char: str = "*"
) -> str:
    """
    Masks a string by replacing some of its characters with a mask character.

    Args:
        s (str): The string to be masked.
        start (int): The number of characters to keep at the beginning of the string.
        end (int): The number of characters to keep at the end of the string.
        mask_char (str): The character to use for masking.

    Returns:
        str: The masked string.

    Raises:
        None
    """
    if s is None or s == "":
        return ""

    if len(s) <= start + end:
        return s[0] + mask_char * (len(s) - 1)

    return (
        s[:start] + mask_char * (len(s) - start - end) + s[-end:]
        if end > 0
        else s[:start] + mask_char * (len(s) - start)
    )


def _get_env_var(var_name: str, critical: bool, mask: bool) -> str:
    """
    Get the value of an environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        critical (bool): Whether or not the function should raise an error if the variable is not set.
        mask (bool): Whether or not to mask the value of the variable in the logs.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the `critical` parameter is True and the environment variable is not set.
    """
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
    """
    A class representing the credentials required to access an Azure Search service.

    Attributes:
        AZURE_SEARCH_SERVICE_ENDPOINT (str): The endpoint URL of the Azure Search service.
        AZURE_SEARCH_ADMIN_KEY (str): The admin key required to access the Azure Search service.
    """

    def __init__(
        self,
        azure_search_service_endpoint: str,
        azure_search_admin_key: str,
    ) -> None:
        self.AZURE_SEARCH_SERVICE_ENDPOINT = azure_search_service_endpoint
        self.AZURE_SEARCH_ADMIN_KEY = azure_search_admin_key

    @classmethod
    def from_env(cls) -> "AzureSearchCredentials":
        """
        Creates an instance of AzureSearchCredentials using environment variables.

        Returns:
            AzureSearchCredentials: An instance of AzureSearchCredentials.
        """
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


class AzureSkillsCredentials:
    """
    A class representing the credentials required to access the skills provided with Azure Cognitive Search.

    Attributes:
        AZURE_LANGUAGE_SERVICE_ENDPOINT (str): The endpoint URL of the Azure Language Detection service.
        AZURE_LANGUAGE_SERVICE_KEY (str): The key required to access the Azure Language Detection service.
    """

    def __init__(
        self,
        azure_language_service_endpoint: str,
        azure_language_service_key: str,
    ) -> None:
        self.AZURE_LANGUAGE_SERVICE_ENDPOINT = azure_language_service_endpoint
        self.AZURE_LANGUAGE_SERVICE_KEY = azure_language_service_key

    @classmethod
    def from_env(cls) -> "AzureSkillsCredentials":
        """
        Creates an instance of AzureSkillsCredentials using environment variables.

        Returns:
            AzureSkillsCredentials: An instance of AzureSkillsCredentials.
        """
        return cls(
            azure_language_service_endpoint=_get_env_var(
                var_name="AZURE_LANGUAGE_SERVICE_ENDPOINT",
                critical=False,
                mask=False,
            ),
            azure_language_service_key=_get_env_var(
                var_name="AZURE_LANGUAGE_SERVICE_KEY",
                critical=False,
                mask=True,
            ),
        )


class AzureMLCredentials:
    """
    A class representing the credentials required to access an Azure Machine Learning workspace.

    Attributes:
        SUBSCRIPTION_ID (str): The subscription ID of the Azure account.
        WORKSPACE_NAME (str): The name of the Azure Machine Learning workspace.
        RESOURCE_GROUP_NAME (str): The name of the resource group containing the Azure Machine Learning workspace.
    """

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
        """
        Creates an instance of AzureMLCredentials using environment variables.

        Returns:
            AzureMLCredentials: An instance of AzureMLCredentials.
        """
        return cls(
            subscription_id=_get_env_var(
                var_name="AML_SUBSCRIPTION_ID",
                critical=False,
                mask=True,
            ),
            workspace_name=_get_env_var(
                var_name="AML_WORKSPACE_NAME",
                critical=False,
                mask=False,
            ),
            resource_group_name=_get_env_var(
                var_name="AML_RESOURCE_GROUP_NAME",
                critical=False,
                mask=False,
            ),
        )


class OpenAICredentials:
    """
    A class to store OpenAI credentials.

    Attributes:
        OPENAI_API_TYPE (str): The type of OpenAI API to use.
        OPENAI_API_KEY (str): The API key for the OpenAI API.
        OPENAI_API_VERSION (str): The version of the OpenAI API to use.
        OPENAI_ENDPOINT (str): The endpoint for the OpenAI API.

    Methods:
        __init__(self, openai_api_type: str, openai_api_key: str, openai_api_version: str, openai_endpoint: str) -> None:
            Initializes the OpenAICredentials object.
        from_env(cls) -> "OpenAICredentials":
            Creates an OpenAICredentials object from environment variables.
        _set_credentials(self) -> None:
            Sets the OpenAI credentials.
    """

    def __init__(
        self,
        openai_api_type: str,
        openai_api_key: str,
        openai_api_version: str,
        openai_endpoint: str,
    ) -> None:
        """
        Initializes the OpenAICredentials object.

        Args:
            openai_api_type (str): The type of OpenAI API to use.
            openai_api_key (str): The API key for the OpenAI API.
            openai_api_version (str): The version of the OpenAI API to use.
            openai_endpoint (str): The endpoint for the OpenAI API.

        Raises:
            ValueError: If some required environment variables are not set.
        """
        # For now we only support Azure Open AI
        self.OPENAI_API_TYPE = "azure"

        if openai_api_version is None:
            raise ValueError(
                f"An OPENAI_API_TYPE of 'azure' requires OPENAI_API_VERSION to"
                f" be set."
            )

        if openai_endpoint is None:
            raise ValueError(
                f"An OPENAI_API_TYPE of 'azure' requires OPENAI_ENDPOINT to be"
                f" set."
            )

        if openai_api_key is None:
            raise ValueError(f"It is required OPENAI_API_KEY to be set.")

        self.OPENAI_API_KEY = openai_api_key
        self.OPENAI_API_VERSION = openai_api_version
        self.OPENAI_ENDPOINT = openai_endpoint

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        """
        Creates an OpenAICredentials object from environment variables.

        Returns:
            OpenAICredentials: The OpenAICredentials object.
        """
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
        self.EMBEDDING_DIMENSIONS = data["embedding_dimension"]
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
        self.SEARCH_RELEVANCY_THRESHOLD = data.get(
            "search_relevancy_threshold", 0.8
        )
        self.DATA_FORMATS = data.get("data_formats", "all")
        self.METRIC_TYPES = data["metric_types"]
        self.LANGUAGE = data.get("language", {})
        self.OpenAICredentials = OpenAICredentials.from_env()
        self.AzureSearchCredentials = AzureSearchCredentials.from_env()
        self.AzureMLCredentials = AzureMLCredentials.from_env()
        self.AzureSkillsCredentials = AzureSkillsCredentials.from_env()

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
