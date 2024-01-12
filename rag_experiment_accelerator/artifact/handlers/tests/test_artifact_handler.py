import os
import pathlib
import shutil
import uuid
import pytest

from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.io.local.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.io.local.writers.jsonl_writer import (
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
    name = "test.jsonl"
    path = f"{temp_dirname}/test.jsonl"
    handler._writer.write(path, data)

    loaded_data = handler.load(name)

    assert loaded_data == [data]


def test_archive(temp_dirname: str):
    data = "This is test data"
    handler = ArtifactHandler(temp_dirname, writer=JsonlWriter(), loader=JsonlLoader())
    filename = "test.jsonl"
    original_path = f"{temp_dirname}/test.jsonl"
    handler._writer.write(original_path, data)

    archive_path = handler.handle_archive(filename)

    # archive dir exists
    assert pathlib.Path(handler.data_location).exists()
    # archive file existså
    assert pathlib.Path(archive_path).exists()
    # original file does not exist
    assert not pathlib.Path(original_path).exists()
