import json
from rag_experiment_accelerator.doc_loader.documentIntelligenceLoader import (
    DocumentIntelligenceLoader,
)
from unittest.mock import patch


class SimplePythonObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def get(self, key, default=None):
        return getattr(self, key, default)


def mock_simple_response(file_name):
    with open(
        f"rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/{file_name}",
        "r",
    ) as f:
        return json.load(f, object_hook=lambda d: SimplePythonObject(**d))


@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths", return_value=["path/to/some/file"],)
@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence")
def test__load(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        'simple_response.json')

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        glob_patterns=["pdf"],
    )

    documents = loader.load()

    assert len(documents) == 1, "No documents were loaded"
    assert (
        documents[0].page_content
        == "This is the Title\n\nSome text\n\nCol 1: Row 1 Col 1, Col 2: Row 1 Col 2, Col 3: Row 1 Col 3 \nCol 1: Row 2 Col 1, Col 2: Row 2 Col 2, Col 3: Row 2 Col 3 \n\nThis is the end."
    )
    assert documents[0].metadata["source"] == "path/to/some/file"
    assert documents[0].metadata["page"] == 0


@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths", return_value=["path/to/some/file"],)
@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence", side_effect=Exception("Error"))
@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._load_with_ocr")
def test_load_with_ocr_is_used_as_fallback(mock_load_with_ocr, _, __):

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        glob_patterns=["pdf"],
    )

    loader.load()

    mock_load_with_ocr.assert_called_once()
    mock_load_with_ocr.assert_called_with('path/to/some/file')


@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths", return_value=["path/to/some/file"],)
@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence")
def test_content_cleaning(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        'simple_response.json')

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        glob_patterns=["pdf"],
        patterns_to_remove=["Ti.*e"]
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "This is the \n\nSome text\n\nCol 1: Row 1 Col 1, Col 2: Row 1 Col 2, Col 3: Row 1 Col 3 \nCol 1: Row 2 Col 1, Col 2: Row 2 Col 2, Col 3: Row 2 Col 3 \n\nThis is the end."
    )


@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths", return_value=["path/to/some/file"],)
@patch("rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence")
def test_table_without_headers(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        'table_without_headers.json')

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        glob_patterns=["pdf"],
        patterns_to_remove=["Ti.*e"]
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "Table without Headers\n===\n\nTesting a table that has no headers\n\nA B \nC D \nE F \nG H \n\nThis is the end."
    )
