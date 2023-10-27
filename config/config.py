import json


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
    def __init__(self, config_filename: str = 'search_config.json') -> None:
        with open(config_filename, 'r') as json_file:
            data = json.load(json_file)

        self.CHUNK_SIZES = data["chunking"]["chunk_size"]
        self.OVERLAP_SIZES = data["chunking"]["overlap_size"]
        self.EMBEDDING_DIMENSIONS = data["embedding_dimension"]
        self.EF_CONSTRUCTIONS = data["efConstruction"]
        self.EF_SEARCHES = data["efSearch"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.CHAT_MODEL_NAME = data["chat_model_name"]
        self.RETRIEVE_NUM_OF_DOCUMENTS = data["retrieve_num_of_documents"]
        self.CROSSENCODER_MODEL = data["crossencoder_model"]
        self.RERANK_TYPE = data["rerank_type"]
        self.LLM_RERANK_THRESHOLD = data["llm_re_rank_threshold"]
        self.CROSSENCODER_AT_K = data["cross_encoder_at_k"]
        self.TEMPERATURE = data["openai_temperature"]
        self.RERANK = data['rerank']
        self.SEARCH_RELEVANCY_THRESHOLD = data.get("search_relevancy_threshold", 0.8)
        self.DATA_FORMATS = data.get("data_formats", "all")
        self.METRIC_TYPES = data["metric_types"]


        with open('querying_config.json', 'r') as json_file:
            data = json.load(json_file)

        self.EVAL_DATA_JSON_FILE_PATH = data["eval_data_json_file_path"]
        self.MAIN_PROMPT_INSTRUCTIONS = data["main_prompt_instruction"]
