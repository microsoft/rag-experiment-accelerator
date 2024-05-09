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

    def keys(self):
        return self.__dict__.keys()


def mock_simple_response(file_name):
    with open(
        f"rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/{file_name}",
        "r",
    ) as f:
        return json.load(f, object_hook=lambda d: SimplePythonObject(**d))


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test__load(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        "simple_response.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
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


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence",
    side_effect=Exception("Error"),
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._load_with_langchain"
)
def test_load_with_langchain_is_used_as_fallback(mock_load_with_langchain, _, __):
    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
    )

    loader.load()

    mock_load_with_langchain.assert_called_once()
    mock_load_with_langchain.assert_called_with("path/to/some/file", "prebuilt-read")


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test_content_cleaning(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        "simple_response.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
        patterns_to_remove=["Ti.*e"],
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "This is the \n\nSome text\n\nCol 1: Row 1 Col 1, Col 2: Row 1 Col 2, Col 3: Row 1 Col 3 \nCol 1: Row 2 Col 1, Col 2: Row 2 Col 2, Col 3: Row 2 Col 3 \n\nThis is the end."
    )


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test_table_without_headers(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        "table_without_headers.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "Table without Headers\n===\n\nTesting a table that has no headers\n\nA B \nC D \nE F \nG H \n\nThis is the end."
    )


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test_document_with_multiple_pages_without_splitting_documents_by_page(
    mock_document_intelligence, _
):
    mock_document_intelligence.return_value = mock_simple_response(
        "multiple_pages.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
        split_documents_by_page=False,
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "Title for page number one Some text for the first page\n\n# Title for page number two\n\nSome text for the 2nd page. Here we also have a table:\n\nName: Alice, Age: 25 \nName: Bob, Age: 32 \n\nTitle for page number three This is the end - at page 3.\n==="
    )
    assert len(documents) == 1


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test_document_with_multiple_pages_with_split_documents_by_page(
    mock_document_intelligence, _
):
    mock_document_intelligence.return_value = mock_simple_response(
        "multiple_pages.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
        split_documents_by_page=True,
    )

    documents = loader.load()

    assert len(documents) == 3
    assert (
        documents[0].page_content
        == "Title for page number one Some text for the first page"
    )
    assert (
        documents[1].page_content
        == "# Title for page number two\n\nSome text for the 2nd page. Here we also have a table:\n\nName: Alice, Age: 25 \nName: Bob, Age: 32 "
    )
    assert (
        documents[2].page_content
        == "Title for page number three This is the end - at page 3.\n==="
    )


@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._get_file_paths",
    return_value=["path/to/some/file"],
)
@patch(
    "rag_experiment_accelerator.doc_loader.documentIntelligenceLoader.DocumentIntelligenceLoader._call_document_intelligence"
)
def test_excluding_paragraphs(mock_document_intelligence, _):
    mock_document_intelligence.return_value = mock_simple_response(
        "multiple_pages.json"
    )

    loader = DocumentIntelligenceLoader(
        path="path",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
        excluded_paragraph_roles=["sectionHeading"],
    )

    documents = loader.load()

    assert (
        documents[0].page_content
        == "Title for page number one Some text for the first page\n\nSome text for the 2nd page. Here we also have a table:\n\nName: Alice, Age: 25 \nName: Bob, Age: 32 \n\nTitle for page number three This is the end - at page 3.\n==="
    )


def test_get_file_paths():
    loader = DocumentIntelligenceLoader(
        path="rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["json"],
    )

    assert set(loader._get_file_paths()) == set(
        [
            "rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/simple_response.json",
            "rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/table_without_headers.json",
            "rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/multiple_pages.json",
        ]
    )


def test_get_file_paths_returns_according_to_glob():
    loader = DocumentIntelligenceLoader(
        path="rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response",
        endpoint="endpoint",
        key="key",
        api_model="prebuilt-layout",
        glob_patterns=["pdf"],
    )

    assert loader._get_file_paths() == []


def test_get_file_paths_works_for_single_files():
    loader = DocumentIntelligenceLoader(
        path="rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/simple_response.json",
        endpoint="endpoint",
        api_model="prebuilt-layout",
        key="key",
    )

    assert loader._get_file_paths() == [
        "rag_experiment_accelerator/doc_loader/tests/test_data/document_intelligence_response/simple_response.json"
    ]
