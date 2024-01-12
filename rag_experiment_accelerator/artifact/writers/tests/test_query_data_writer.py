import os
import shutil
import uuid
import pytest

from rag_experiment_accelerator.artifact.loaders.query_output_loader import (
    QueryOutputLoader,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.artifact.writers.query_output_writer import (
    QueryOutputWriter,
)


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_handle_archive(temp_dirname: str):
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
    writer = QueryOutputWriter(temp_dirname)
    for d in test_data:
        writer.save(d, index_name)

    dest = writer.handle_archive(index_name)

    assert os.path.exists(writer.archive_dir)
    assert os.path.exists(dest)
    assert not os.path.exists(writer.get_output_filepath(index_name))


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
    writer = QueryOutputWriter(temp_dirname)
    for d in test_data:
        writer.save(d, index_name)

    loader = QueryOutputLoader(temp_dirname)
    loaded_data = loader.load_all(index_name)
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
