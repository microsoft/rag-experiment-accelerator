import json
import openai
from rag_experiment_accelerator.llm.factory import EmbeddingModelFactory
from rag_experiment_accelerator.llm.base import EmbeddingModel
from rag_experiment_accelerator.utils.auth import OpenAICredentials
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.utils import get_env_var

logger = get_logger(__name__)




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
            azure_search_service_endpoint=get_env_var(
                var_name="AZURE_SEARCH_SERVICE_ENDPOINT",
                critical=False,
                mask=False,
            ),
            azure_search_admin_key=get_env_var(
                var_name="AZURE_SEARCH_ADMIN_KEY",
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
            subscription_id=get_env_var(
                var_name="AML_SUBSCRIPTION_ID",
                critical=False,
                mask=True,
            ),
            workspace_name=get_env_var(
                var_name="AML_WORKSPACE_NAME",
                critical=False,
                mask=False,
            ),
            resource_group_name=get_env_var(
                var_name="AML_RESOURCE_GROUP_NAME",
                critical=False,
                mask=False,
            ),
        )




class Config:
    """
    A class for storing configuration settings for the RAG Experiment Accelerator.

    Parameters:
        config_filename (str): The name of the JSON file containing configuration settings. Default is 'search_config.json'.

    Attributes:
        CHUNK_SIZES (list[int]): A list of integers representing the chunk sizes for chunking documents.
        OVERLAP_SIZES (list[int]): A list of integers representing the overlap sizes for chunking documents.
        EMBEDDING_DIMENSIONS (list[int]): The number of dimensions to use for document embeddings.
        EF_CONSTRUCTIONS (list[int]): The number of ef_construction to use for HNSW index.
        EF_SEARCHES (list[int]): The number of ef_search to use for HNSW index.
        NAME_PREFIX (str): A prefix to use for the names of saved models.
        SEARCH_VARIANTS (list[str]): A list of search types to use.
        CHAT_MODEL_NAME (str): The name of the chat model to use.
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
        self.EF_CONSTRUCTIONS = data["ef_construction"]
        self.EF_SEARCHES = data["ef_search"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.CHAT_MODEL_NAME = data.get("chat_model_name", None)
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

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = data.get("embedding_models", [])
        for model in embedding_model_config:
            self.embedding_models.append(EmbeddingModelFactory.create(model.get("type"), model.get("model_name"), model.get("dimension"), self.OpenAICredentials))

        self._check_deployment()

        with open("querying_config.json", "r") as json_file:
            data = json.load(json_file)

        self.EVAL_DATA_JSON_FILE_PATH = data["eval_data_json_file_path"]
        self.MAIN_PROMPT_INSTRUCTIONS = data["main_prompt_instruction"]


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

        This function checks if the OpenAI API type and chat model name are set,
        and then tries to retrieve the model with specified tags.

        """
        # for embedding_model in self.embedding_models:
        #     embedding_model.try_retrieve_model()
        #     logger.info(f"Model {embedding_model.model_name} is ready for use.")

        # self.chat_model.try_retrieve_model()
        # logger.info(f"Model {self.CHAT_MODEL_NAME} is ready for use.")
        if self.OpenAICredentials.OPENAI_API_TYPE is not None:
            if self.CHAT_MODEL_NAME is not None:
                self._try_retrieve_model(
                    self.CHAT_MODEL_NAME,
                    tags=["chat_completion", "inference"],
                )
                logger.info(f"Model {self.CHAT_MODEL_NAME} is ready for use.")
            # for embedding_model in self.embedding_models:
            #     self.try_retrieve_model(
            #         embedding_model.model_name,
            #         tags=["embeddings", "inference"],
            #     )
            #     logger.info(f"Model {embedding_model.model_name} is ready for use.")
            # if self.EMBEDDING_MODEL_NAME is not None:
            #     self._try_retrieve_model(
            #         self.EMBEDDING_MODEL_NAME,
            #         tags=["embeddings", "inference"],
            #     )
            #     logger.info(f"Model {self.EMBEDDING_MODEL_NAME} is ready for use.")
