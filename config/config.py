import json


class Config:
    def __init__(self) -> None:
        with open('search_config.json', 'r') as json_file:
            data = json.load(json_file)

        self.CHUNK_SIZES = data["chunking"]["chunk_size"]
        self.OVERLAP_SIZES = data["chunking"]["overlap_size"]
        self.EMBEDDING_DIMENSIONS = data["embedding_dimension"]
        self.EF_CONSTRUCTIONS  = data["efConstruction"]
        self.EF_SEARCH = data["efsearch"]
        self.NAME_PREFIX = data["name_prefix"]
        self.SEARCH_VARIANTS = data["search_types"]
        self.CHAT_MODEL_NAME=data["chat_model_name"]
        self.RETRIEVE_NUM_OF_DOCUMENTS = data["retrieve_num_of_documents"]
        self.CROSSENCODER_MODEL = data["crossencoder_model"]
        self.RERANK_TYPE = data["rerank_type"]
        self.LLM_RERANK_THRESHOLD = data["llm_re_rank_threshold"]
        self.CROSSENCODER_AT_K = data["cross_encoder_at_k"]
        self.TEMPERATURE = data["openai_temperature"]
        self.RERANK = data['rerank']
        self.SEARCH_RELEVANCY_THRESHOLD = data.get("search_relevancy_threshold", 0.8)
