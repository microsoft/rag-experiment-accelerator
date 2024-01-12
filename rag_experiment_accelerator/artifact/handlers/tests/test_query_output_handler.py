import os
import shutil
import tempfile
import uuid
import pytest
from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)

from rag_experiment_accelerator.artifact.models.query_output import QueryOutput


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


@pytest.fixture()
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_archive(temp_dirname: str):
    index_name = "index_name"
    test_data = [
        QueryOutput(
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
        ),
        QueryOutput(
            rerank="rerank2",
            rerank_type="rerank_type2",
            crossencoder_model="cross_encoder_model2",
            llm_re_rank_threshold=2,
            retrieve_num_of_documents=2,
            crossencoder_at_k=2,
            question_count=2,
            actual="actual2",
            expected="expected2",
            search_type="search_type2",
            search_evals=[],
            context="context2",
        ),
    ]
    handler = QueryOutputHandler(temp_dirname)
    for d in test_data:
        handler.save(d, index_name)

    dest = handler.archive(index_name)

    assert os.path.exists(handler.archive_dir)
    assert os.path.exists(dest)
    assert not os.path.exists(handler.get_output_filepath(index_name))


def test_save(temp_dirname: str):
    test_data = [
        QueryOutput(
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
        ),
        QueryOutput(
            rerank="rerank2",
            rerank_type="rerank_type2",
            crossencoder_model="cross_encoder_model2",
            llm_re_rank_threshold=2,
            retrieve_num_of_documents=2,
            crossencoder_at_k=2,
            question_count=2,
            actual="actual2",
            expected="expected2",
            search_type="search_type2",
            search_evals=[],
            context="context2",
        ),
    ]
    index_name = "index_name"
    handler = QueryOutputHandler(temp_dirname)
    for d in test_data:
        handler.save(d, index_name)

    loaded_data = handler.load(index_name)
    for i, d in enumerate(loaded_data):
        assert d.rerank == test_data[i].rerank
        assert d.rerank_type == test_data[i].rerank_type
        assert d.crossencoder_model == test_data[i].crossencoder_model
        assert d.llm_re_rank_threshold == test_data[i].llm_re_rank_threshold
        assert d.retrieve_num_of_documents == test_data[i].retrieve_num_of_documents
        assert d.crossencoder_at_k == test_data[i].crossencoder_at_k
        assert d.question_count == test_data[i].question_count
        assert d.actual == test_data[i].actual
        assert d.expected == test_data[i].expected
        assert d.search_type == test_data[i].search_type
        assert d.search_evals == test_data[i].search_evals
        assert d.context == test_data[i].context


def test_load(temp_dir):
    test_data = [
        QueryOutput(
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
        ),
        QueryOutput(
            rerank="rerank2",
            rerank_type="rerank_type2",
            crossencoder_model="cross_encoder_model2",
            llm_re_rank_threshold=2,
            retrieve_num_of_documents=2,
            crossencoder_at_k=2,
            question_count=2,
            actual="actual2",
            expected="expected2",
            search_type="search_type2",
            search_evals=[],
            context="context2",
        ),
    ]
    index_name = "index_name"
    filename = "test.jsonl"
    path = f"{temp_dir}/{filename}"
    # write the data
    handler = QueryOutputHandler(path)
    for d in test_data:
        handler.save(d, index_name)

    # load the data
    handler = QueryOutputHandler(path)
    loaded_data = handler.load(index_name)

    # assertions
    for i, d in enumerate(loaded_data):
        assert d.rerank == test_data[i].rerank
        assert d.rerank_type == test_data[i].rerank_type
        assert d.crossencoder_model == test_data[i].crossencoder_model
        assert d.llm_re_rank_threshold == test_data[i].llm_re_rank_threshold
        assert d.retrieve_num_of_documents == test_data[i].retrieve_num_of_documents
        assert d.crossencoder_at_k == test_data[i].crossencoder_at_k
        assert d.question_count == test_data[i].question_count
        assert d.actual == test_data[i].actual
        assert d.expected == test_data[i].expected
        assert d.search_type == test_data[i].search_type
        assert d.search_evals == test_data[i].search_evals
        assert d.context == test_data[i].context
