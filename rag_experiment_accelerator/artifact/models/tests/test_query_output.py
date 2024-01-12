from rag_experiment_accelerator.artifact.models.query_output import QueryOutput


def test_to_dict():
    output = QueryOutput(
        rerank="rerank",
        rerank_type="rerank_type",
        crossencoder_model="crossencoder_model",
        llm_re_rank_threshold="llm_re_rank_threshold",
        retrieve_num_of_documents="retrieve_num_of_documents",
        crossencoder_at_k="crossencoder_at_k",
        question_count="question_count",
        actual="actual",
        expected="expected",
        search_type="search_type",
        search_evals="search_evals",
        context="context",
    )
    output_dict = output.to_dict()
    assert output_dict["rerank"] == output.rerank
    assert output_dict["rerank_type"] == output.rerank_type
    assert output_dict["crossencoder_model"] == output.crossencoder_model
    assert output_dict["llm_re_rank_threshold"] == output.llm_re_rank_threshold
    assert output_dict["retrieve_num_of_documents"] == output.retrieve_num_of_documents
    assert output_dict["crossencoder_at_k"] == output.crossencoder_at_k
    assert output_dict["question_count"] == output.question_count
    assert output_dict["actual"] == output.actual
    assert output_dict["expected"] == output.expected
    assert output_dict["search_type"] == output.search_type
    assert output_dict["search_evals"] == output.search_evals
    assert output_dict["context"] == output.context
