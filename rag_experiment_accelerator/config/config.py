import json
import os
import openai
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def _mask_string(s: str, start: int = 2, end: int = 2, mask_char: str = "*") -> str:
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
            ValueError: If openai_api_type is not 'azure' or 'open_ai'.
        """
        if openai_api_type is not None and openai_api_type not in ["azure", "open_ai"]:
            logger.critical("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")
            raise ValueError("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")

        if openai_api_type == "azure" and openai_api_version is None:
            raise ValueError(
                f"An OPENAI_API_TYPE of 'azure' requires OPENAI_API_VERSION to be set."
            )

        if openai_api_type == "azure" and openai_endpoint is None:
            raise ValueError(
                f"An OPENAI_API_TYPE of 'azure' requires OPENAI_ENDPOINT to be set."
            )

        self.OPENAI_API_TYPE = openai_api_type
        self.OPENAI_API_KEY = openai_api_key
        self.OPENAI_API_VERSION = openai_api_version
        self.OPENAI_ENDPOINT = openai_endpoint

        self._set_credentials()

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

    def _set_credentials(self) -> None:
        """
        Sets the OpenAI credentials.
        """
        if self.OPENAI_API_TYPE is not None:
            openai.api_type = self.OPENAI_API_TYPE
            openai.api_key = self.OPENAI_API_KEY
            logger.info(f"OpenAI API key set to {_mask_string(openai.api_key)}")

            if self.OPENAI_API_TYPE == "azure":
                openai.api_version = self.OPENAI_API_VERSION
                openai.api_base = self.OPENAI_ENDPOINT


# imported here to avoid circular imports - we should think about moving all Credentials to its own file
from rag_experiment_accelerator.llm.embeddings.factory import EmbeddingModelFactory
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel


class Config:
    """
    A class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'search_config.json'.

    Attributes:
        CHUNK_SIZES (list[int]): A list of integers representing the chunk sizes for chunking documents.
        OVERLAP_SIZES (list[int]): A list of integers representing the overlap sizes for chunking documents.
        EF_CONSTRUCTIONS (list[int]): The number of ef_construction to use for HNSW index.
        EF_SEARCHES (list[int]): The number of ef_search to use for HNSW index.
        NAME_PREFIX (str): A prefix to use for the names of saved models.
        SEARCH_VARIANTS (list[str]): A list of search types to use.
        CHAT_MODEL_NAME (str): The name of the chat model to use.
        EVAL_MODEL_NAME (str): The name of the chat model to use for prod.
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
        OpenAICredentials (OpenAICredentials): OpenAI credentials.
        AzureSearchCredentials (AzureSearchCredentials): Azure Search credentials.
        AzureMLCredentials (AzureMLCredentials): Azure ML credentials.
        emebdding_models (list[EmbeddingModel]): a list of emebedding models to use for document embeddings.
    """

    def __init__(self, config_filename: str = "search_config.json") -> None:
        with open(config_filename, "r") as json_file:
            data = json.load(json_file)

        self.CHUNK_SIZES = data["chunking"]["chunk_size"]
        self.OVERLAP_SIZES = data["chunking"]["overlap_size"]
        self.EF_CONSTRUCTIONS = data["ef_construction"]
        self.EF_SEARCHES = data["ef_search"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.CHAT_MODEL_NAME = data.get("chat_model_name", None)
        self.EVAL_MODEL_NAME = data.get("eval_model_name", None)
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
        self.EVAL_DATA_JSONL_FILE_PATH = data["eval_data_jsonl_file_path"]
        self.OpenAICredentials = OpenAICredentials.from_env()
        self.AzureSearchCredentials = AzureSearchCredentials.from_env()
        self.AzureMLCredentials = AzureMLCredentials.from_env()
        self.AzureSkillsCredentials = AzureSkillsCredentials.from_env()

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = data.get("embedding_models", [])
        for model in embedding_model_config:
            self.embedding_models.append(
                EmbeddingModelFactory.create(
                    model.get("type"),
                    model.get("model_name"),
                    model.get("dimension"),
                    self.OpenAICredentials,
                )
            )

        self._check_deployment()

        with open("prompt_config.json", "r") as json_file:
            data = json.load(json_file)

        self.MAIN_PROMPT_INSTRUCTION = data["main_prompt_instruction"]

    def _try_retrieve_model(self, model_name: str, tags: list[str]) -> openai.Model:
        """
        Tries to retrieve a specified model from OpenAI.

        Args:
            model_name (str): The name of the model to retrieve.
            tags (list[str]): A list of capability tags to check for.
        Returns:
            openai.Model: The retrieved model object if successful.

        Raises:
            ValueError: If the model is not ready or does not have the required capabilities.
            openai.error.InvalidRequestError: If the model does not exist.
        """
        try:
            model = openai.Model.retrieve(model_name)

            # For non-azure models we can't retrieve status and capabilities
            if self.OpenAICredentials.OPENAI_API_TYPE != "azure":
                return model
            if model["status"] != "succeeded":
                logger.critical(f"Model {model_name} is not ready.")
                raise ValueError(f"Model {model_name} is not ready.")
            for tag in tags:
                if not model["capabilities"][tag]:
                    logger.critical(
                        f"Model {model_name} does not have the {tag} capability."
                    )
                    raise ValueError(
                        f"Model {model_name} does not have the {tag} capability."
                    )
            return model
        except openai.error.InvalidRequestError as e:
            logger.critical(f"Model {model_name} does not exist.")
            raise ValueError(f"Model {model_name} does not exist.")

    def _check_deployment(self):
        """
        Checks the deployment environment.

        This function checks if the embedding models and chat model are ready for use.
        It tries to retrieve the embedding models and the chat model with specified tags.
        """

        if self.OpenAICredentials.OPENAI_API_TYPE is not None:
            if self.CHAT_MODEL_NAME is not None:
                self._try_retrieve_model(
                    self.CHAT_MODEL_NAME,
                    tags=["chat_completion", "inference"],
                )
                logger.info(f"Model {self.CHAT_MODEL_NAME} is ready for use.")
            for embedding_model in self.embedding_models:
                embedding_model.try_retrieve_model()
                logger.info(f"Model {embedding_model.model_name} is ready for use.")
