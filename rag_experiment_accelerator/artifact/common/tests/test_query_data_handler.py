from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryOutputHandler,
)


def test_get_output_filename():
    index_name = "index_name"
    handler = QueryOutputHandler("")
    output_filename = handler.get_output_filename(index_name)
    assert output_filename == f"eval_output_{index_name}.jsonl"


def test_get_output_filepath():
    output_dir = "output"
    index_name = "index_name"
    handler = QueryOutputHandler(output_dir)
    output_filename = handler.get_output_filepath(index_name)
    assert output_filename == f"{output_dir}/{handler.get_output_filename(index_name)}"
