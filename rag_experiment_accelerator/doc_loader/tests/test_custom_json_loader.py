from langchain.document_loaders import JSONLoader

from rag_experiment_accelerator.doc_loader.customJsonLoader import \
    CustomJSONLoader
from rag_experiment_accelerator.doc_loader.structuredLoader import \
    load_structured_files


# The output of our customJsonLoader should be equal to the output of langchains JSONLoader
def test_load_json_files():
    folder_path = "./data/"
    chunk_size = 1000
    overlap_size = 200

    original_doc = load_structured_files(
        file_format="JSON",
        language=None,
        loader=JSONLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=["json"],
        loader_kwargs={
            "jq_schema": "[.[] | {content: .content, title: .title}]",
            "text_content": False,
        },
    )

    custom_doc = load_structured_files(
        file_format="JSON",
        language=None,
        loader=CustomJSONLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=["json"],
        loader_kwargs={
            "jq_schema": "[.[] | {content: .content, title: .title}]",
            "text_content": False,
        },
    )

    assert len(original_doc) == len(custom_doc)

    assert original_doc[0].page_content == custom_doc[0].page_content
    assert original_doc[0].metadata == custom_doc[0].metadata
