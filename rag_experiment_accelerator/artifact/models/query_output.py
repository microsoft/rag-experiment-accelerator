from rag_experiment_accelerator.artifact.models.artifact import Artifact


class QueryOutput(Artifact):
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
