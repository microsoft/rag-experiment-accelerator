import json
import os
import shutil
import tempfile
import pytest

from rag_experiment_accelerator.io.local.loaders.jsonl_loader import JsonlLoader


@pytest.fixture()
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_loads(temp_dir: str):
    test_data = {"test": {"test1": 1, "test2": 2}}
    # write the file
    path = f"{temp_dir}/test.jsonl"
    with open(path, "a") as file:
        file.write(json.dumps(test_data) + "\n")

    # load the file
    loader = JsonlLoader()
    loaded_data = loader.load(path)

    assert loaded_data == [test_data]


def test_loads_raises_file_not_found(temp_dir: str):
    path = f"{temp_dir}/non-existsing-file.jsonl"
    loader = JsonlLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(path)


def test_can_handle_true():
    path = "test.jsonl"
    loader = JsonlLoader()
    assert loader.can_handle(path) is True


def test_can_handle_false():
    path = "test.txt"
    loader = JsonlLoader()
    assert loader.can_handle(path) is False
