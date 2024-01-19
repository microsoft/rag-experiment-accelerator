import os
import shutil
import tempfile
import pytest

from rag_experiment_accelerator.io.local.base import LocalIOBase


@pytest.fixture()
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_exists_true(temp_dir: str) -> bool:
    loader = LocalIOBase()
    assert loader.exists(temp_dir) is True


def test_exists_false() -> bool:
    path = "/tmp/non-existing-file"
    loader = LocalIOBase()
    assert loader.exists(path) is False
