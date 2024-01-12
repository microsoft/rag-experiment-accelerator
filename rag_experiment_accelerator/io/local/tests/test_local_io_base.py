import os
import shutil
import uuid

import pytest

from rag_experiment_accelerator.io.local.base import LocalIOBase


@pytest.fixture()
def temp_file():
    dir = "/tmp/" + uuid.uuid4().__str__()
    os.makedirs(dir)
    filename = "test.txt"
    path = dir + "/" + filename
    with open(path, "w") as f:
        f.write("test")
    yield path
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_exists_true(temp_file: str) -> bool:
    loader = LocalIOBase()
    assert loader.exists(temp_file) is True


def test_exists_false() -> bool:
    path = "/tmp/non-existing-file"
    loader = LocalIOBase()
    assert loader.exists(path) is False
