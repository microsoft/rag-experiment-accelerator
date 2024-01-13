from unittest.mock import Mock

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput


def test_archive():
    mock_writer = Mock()
    mock_loader = Mock()
    mock_writer.exists.return_value = True
    index_name = "index_name"
    data_location = "data_location"
    handler = QueryOutputHandler(
        data_location=data_location, writer=mock_writer, loader=mock_loader
    )
    dest = handler.handle_archive_by_index(index_name)

    src = f"{data_location}/eval_output_{index_name}.jsonl"
    mock_writer.copy.assert_called_once_with(src, dest)
    mock_writer.delete.assert_called_once_with(src)


def test_get_output_path():
    index_name = "index_name"
    dir = "/tmp"
    handler = QueryOutputHandler(dir)
    dest = handler.get_output_path(index_name)
    name = handler._get_output_name(index_name)
    assert dest == f"{dir}/{name}"


def test__get_output_name():
    index_name = "index_name"
    dir = "/tmp"
    handler = QueryOutputHandler(dir)
    name = handler._get_output_name(index_name)
    assert name == f"eval_output_{index_name}.jsonl"


def test_save():
    mock_writer = Mock()
    index_name = "index_name"
    test_data = QueryOutput(
        rerank="rerank1",
        rerank_type="rerank_type1",
        crossencoder_model="cross_encoder_model1",
        llm_re_rank_threshold=1,
        retrieve_num_of_documents=1,
        crossencoder_at_k=2,
        question_count=1,
        actual="actual1",
        expected="expected1",
        search_type="search_type1",
        search_evals=[],
        context="context1",
    )

    handler = QueryOutputHandler(data_location="data_location", writer=mock_writer)
    handler.save(test_data, index_name)

    path = handler.get_output_path(index_name)
    mock_writer.write.assert_called_once_with(path, test_data.__dict__)


def test_load():
    data = QueryOutput(
        rerank="rerank1",
        rerank_type="rerank_type1",
        crossencoder_model="cross_encoder_model1",
        llm_re_rank_threshold=1,
        retrieve_num_of_documents=1,
        crossencoder_at_k=1,
        question_count=1,
        actual="actual1",
        expected="expected1",
        search_type="search_type1",
        search_evals=[],
        context="context1",
    )

    mock_loader = Mock()
    mock_loader.load.return_value = [data.__dict__]
    index_name = "index_name"

    handler = QueryOutputHandler(data_location="data_location", loader=mock_loader)
    loaded_data = handler.load(index_name)
    assert len(loaded_data) == 1
    for d in loaded_data:
        assert d.rerank == data.rerank
        assert d.rerank_type == data.rerank_type
        assert d.crossencoder_model == data.crossencoder_model
        assert d.llm_re_rank_threshold == data.llm_re_rank_threshold
        assert d.retrieve_num_of_documents == data.retrieve_num_of_documents
        assert d.crossencoder_at_k == data.crossencoder_at_k
        assert d.question_count == data.question_count
        assert d.actual == data.actual
        assert d.expected == data.expected
        assert d.search_type == data.search_type
        assert d.search_evals == data.search_evals
        assert d.context == data.context
