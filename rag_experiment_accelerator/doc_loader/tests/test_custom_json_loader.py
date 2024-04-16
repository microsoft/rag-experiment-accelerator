import os
import pytest

from rag_experiment_accelerator.doc_loader.customJsonLoader import (
    CustomJSONLoader,
)
from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)


def test_load_json_files():
    keys_to_load = ["content", "title"]
    expected_content = [
        {"content": "This is the content for item 1.", "title": "Title TEST 1"},
        {"content": "This is the content for item 2.", "title": "Title 2"},
        {"content": "This is the content for item 3.", "title": "Title 3"},
        {"content": "This is the content for item 4.", "title": "Title 4"},
        {"content": "This is the content for item 5.", "title": "Title 5"},
        {"content": "This is the content for item 6.", "title": "Title 6"},
    ]

    docs = load_structured_files(
        file_format="JSON",
        language=None,
        loader=CustomJSONLoader,
        file_paths=[
            "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.valid.json"
        ],
        chunk_size=1000,
        overlap_size=200,
        loader_kwargs={
            "keys_to_load": keys_to_load,
        },
    )
    doc = list(docs[0].values())[0]
    assert doc["content"] == str(expected_content)
    assert doc["metadata"]["source"].endswith(
        "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.valid.json"
    )


def test_load_json_files_raises_invalid_keys():
    keys_to_load = ["content", "title"]
    with pytest.raises(ValueError) as exec_info:
        load_structured_files(
            file_format="JSON",
            language=None,
            loader=CustomJSONLoader,
            file_paths=[
                os.path.abspath(
                    "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.bad.invalid_keys.json"
                )
            ],
            chunk_size=1000,
            overlap_size=200,
            loader_kwargs={
                "keys_to_load": keys_to_load,
            },
        )

    file_path = os.path.abspath(
        "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.bad.invalid_keys.json"
    )
    assert (
        str(exec_info.value)
        == f"JSON file at path {file_path} must contain the field 'content'"
    )


def test_load_json_files_raises_not_a_list():
    keys_to_load = ["content", "title"]
    loader = CustomJSONLoader
    with pytest.raises(ValueError) as exec_info:
        load_structured_files(
            file_format="JSON",
            language=None,
            loader=loader,
            file_paths=[
                os.path.abspath(
                    "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.bad.not_a_list.json"
                )
            ],
            chunk_size=1000,
            overlap_size=200,
            loader_kwargs={
                "keys_to_load": keys_to_load,
            },
        )

    file_path = os.path.abspath(
        "rag_experiment_accelerator/doc_loader/tests/test_data/json/data.bad.not_a_list.json"
    )
    assert (
        str(exec_info.value)
        == f"JSON file at path: {file_path} must be a list of object and expects each object to contain the fields ['content', 'title']"
    )
