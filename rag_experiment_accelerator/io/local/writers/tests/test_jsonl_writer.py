import os
import shutil
import tempfile

import pytest

from rag_experiment_accelerator.io.local.writers.jsonl_writer import (
    JsonlWriter,
)


@pytest.fixture()
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test__write_file(temp_dir: str):
    # set up
    data = {"test": "test"}
    path = temp_dir + "/test.jsonl"

    # write the file
    writer = JsonlWriter()
    writer._write_file(path, data)

    # check file was written and contains the correct data
    with open(path) as file:
        assert file.readline() == '{"test": "test"}\n'
