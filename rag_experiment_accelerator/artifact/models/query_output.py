class QueryOutput:
    """
    Represents the output of a query.

    Attributes:
        rerank (bool): Indicates whether reranking is enabled.
        rerank_type (str): The type of reranking.
        cross_encoder_model (str): The model used for cross-encoding.
        llm_rerank_threshold (int): The threshold for reranking using LLM.
        retrieve_num_of_documents (int): The number of documents to retrieve.
        crossencoder_at_k (int): The value of k for cross-encoder.
        question_count (int): The count of questions.
        actual (str): The actual output.
        expected (str): The expected output.
        search_type (str): The type of search.
        search_evals (list): The evaluations for search.
        context (str): The qna context of the query.
        retrieved_contexts (list): The list of retrieved contexts of the query.
        question (str): The question of the query.
    """

    def __init__(
        self,
        rerank: bool,
        rerank_type: str,
        cross_encoder_model: str,
        llm_rerank_threshold: int,
        retrieve_num_of_documents: int,
        crossencoder_at_k: int,
        question_count: int,
        actual: str,
        expected: str,
        search_type: str,
        search_evals: list,
        context: str,
        retrieved_contexts: list,
        question: str,
    ):
        self.rerank = rerank
        self.rerank_type = rerank_type
        self.cross_encoder_model = cross_encoder_model
        self.llm_rerank_threshold = llm_rerank_threshold
        self.retrieve_num_of_documents = retrieve_num_of_documents
        self.crossencoder_at_k = crossencoder_at_k
        self.question_count = question_count
        self.actual = actual
        self.expected = expected
        self.search_type = search_type
        self.search_evals = search_evals
        self.context = context
        self.retrieved_contexts = retrieved_contexts
        self.question = question
