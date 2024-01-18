class QueryOutput:
    """
    Represents the output of a query.

    Attributes:
        rerank (bool): Indicates whether reranking is enabled.
        rerank_type (str): The type of reranking.
        crossencoder_model (str): The model used for cross-encoding.
        llm_re_rank_threshold (int): The threshold for reranking using LLM.
        retrieve_num_of_documents (int): The number of documents to retrieve.
        crossencoder_at_k (int): The value of k for cross-encoder.
        question_count (int): The count of questions.
        actual (str): The actual output.
        expected (str): The expected output.
        search_type (str): The type of search.
        search_evals (list): The evaluations for search.
        context (str): The context of the query.
        question (str): The question of the query.
    """

    def __init__(
        self,
        rerank: bool,
        rerank_type: str,
        crossencoder_model: str,
        llm_re_rank_threshold: int,
        retrieve_num_of_documents: int,
        crossencoder_at_k: int,
        question_count: int,
        actual: str,
        expected: str,
        search_type: str,
        search_evals: list,
        context: str,
        question: str,
    ):
        self.rerank = rerank
        self.rerank_type = rerank_type
        self.crossencoder_model = crossencoder_model
        self.llm_re_rank_threshold = llm_re_rank_threshold
        self.retrieve_num_of_documents = retrieve_num_of_documents
        self.crossencoder_at_k = crossencoder_at_k
        self.question_count = question_count
        self.actual = actual
        self.expected = expected
        self.search_type = search_type
        self.search_evals = search_evals
        self.context = context
        self.question = question
