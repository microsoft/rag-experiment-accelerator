import os
import pathlib
import shutil
import uuid
import pytest

from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)

from rag_experiment_accelerator.loaders.local.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.writers.local.jsonl_writer import (
    JsonlWriter,
)


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_loads(temp_dirname: str):
    # # write artifacts to a file
    data = "This is test data"
    handler = ArtifactHandler(temp_dirname, writer=JsonlWriter(), loader=JsonlLoader())
    filename = "test.jsonl"
    filepath = f"{temp_dirname}/test.jsonl"
    handler._writer.write(filepath, data)

    loaded_data = handler.load(filename)

    assert loaded_data == [data]


def test_archive(temp_dirname: str):
    data = "This is test data"
    handler = ArtifactHandler(temp_dirname, writer=JsonlWriter(), loader=JsonlLoader())
    filename = "test.jsonl"
    original_filepath = f"{temp_dirname}/test.jsonl"
    handler._writer.write(original_filepath, data)

    archive_filepath = handler.archive(filename)

    # archive dir exists
    assert pathlib.Path(handler.archive_dir).exists()
    # archive file existså
    assert pathlib.Path(archive_filepath).exists()
    # original file does not exist
    assert not pathlib.Path(original_filepath).exists()
