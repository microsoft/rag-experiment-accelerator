import json
import openai
from rag_experiment_accelerator.llm.embeddings.factory import EmbeddingModelFactory
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingModel
from rag_experiment_accelerator.config.auth import OpenAICredentials, AzureSearchCredentials, AzureMLCredentials
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


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
        LANGUAGE (dict): Language settings.
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

        self.embedding_models: list[EmbeddingModel] = []
        embedding_model_config = data.get("embedding_models", [])
        for model in embedding_model_config:
            self.embedding_models.append(EmbeddingModelFactory.create(model.get("type"), model.get("model_name"), model.get("dimension"), self.OpenAICredentials))

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
