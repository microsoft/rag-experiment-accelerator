from unittest.mock import patch

import pytest

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput


@patch(
    "rag_experiment_accelerator.artifact.handlers.query_output_handler.ArtifactHandler.handle_archive"
)
def test_handle_archive_by_index(mock_artifact_handler_handle_archive):
    index_name = "index_name"
    experiment_name = "experiment_name"
    job_name = "job_name"
    data_location = "data_location"
    handler = QueryOutputHandler(data_location=data_location)

    handler.handle_archive_by_index(index_name, experiment_name, job_name)

    output_filename = handler._get_output_name(index_name, experiment_name, job_name)
    mock_artifact_handler_handle_archive.assert_called_once_with(output_filename)


def test_get_output_path():
    index_name = "index_name"
    experiment_name = "experiment_name"
    job_name = "job_name"
    dir = "/tmp"
    handler = QueryOutputHandler(dir)
    dest = handler.get_output_path(index_name, experiment_name, job_name)
    name = handler._get_output_name(index_name, experiment_name, job_name)
    assert dest == f"{dir}/{name}"


def test__get_output_name():
    index_name = "index_name"
    experiment_name = "experiment_name"
    job_name = "job_name"

    dir = "/tmp"
    handler = QueryOutputHandler(dir)
    name = handler._get_output_name(index_name, experiment_name, job_name)
    assert name == f"eval_output_{index_name}_{experiment_name}_{job_name}.jsonl"


@patch(
    "rag_experiment_accelerator.artifact.handlers.query_output_handler.ArtifactHandler.save_dict"
)
def test_save(mock_artifact_handler_save_dict):
    index_name = "index_name"
    experiment_name = "experiment_name"
    job_name = "job_name"

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
        question="question1",
    )

    handler = QueryOutputHandler(data_location="data_location")
    handler.save(test_data, index_name, experiment_name, job_name)

    name = handler._get_output_name(index_name, experiment_name, job_name)
    handler.save_dict.assert_called_once_with(test_data.__dict__, name)


@patch(
    "rag_experiment_accelerator.artifact.handlers.query_output_handler.ArtifactHandler.load"
)
def test_load(mock_artifact_handler_load):
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
        question="question1",
    )

    mock_artifact_handler_load.return_value = [data.__dict__, data.__dict__]
    index_name = "index_name"
    experiment_name = "experiment_name"
    job_name = "job_name"

    handler = QueryOutputHandler(data_location="data_location")
    loaded_data = handler.load(index_name, experiment_name, job_name)

    assert len(loaded_data) == 2
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


@patch(
    "rag_experiment_accelerator.artifact.handlers.query_output_handler.ArtifactHandler.load"
)
def test_load_raises_when_loaded_data_not_dict(mock_artifact_handler_load):
    mock_artifact_handler_load.return_value = ["this is not a dict"]
    index_name = "index_name"

    handler = QueryOutputHandler(data_location="data_location")

    with pytest.raises(TypeError):
        handler.load(index_name)
