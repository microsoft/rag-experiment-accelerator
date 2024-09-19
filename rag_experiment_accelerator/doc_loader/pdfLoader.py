from typing import List
import uuid
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def preprocess_pdf_content(content: str):
    """
    Preprocess the content extracted from a PDF file.
    This function performs the following preprocessing steps on the input content:
    1. Replaces multiple consecutive newline characters ('\\n') with a single newline character.
    2. Removes all remaining newline characters.
    3. Removes Unicode escape sequences in the format '\\uXXXX' where X is a hexadecimal digit.
    4. Converts the content to lowercase.
    Args:
        content (str): The content extracted from the PDF file.
    Returns:
        str: The preprocessed content.
    Example:
        content = "Hello\\n\\nWorld\\n\\u1234 OpenAI"
        preprocessed_content = preprocess_pdf_content(content)
        print(preprocessed_content)
        # Output: "hello world openai"
    """

    content = re.sub(r"\n{2,}", "\n", content)
    content = re.sub(r"\n{1,}", "", content)
    content = re.sub(r"\\u[0-9a-fA-F]{4}", "", content)
    content = content.lower()

    return content


def _load_pdf_with_pypdf(file_path: str) -> List[Document]:
    logger.debug("Loading PDF file with PyPDF: %s", file_path)
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()


def _load_pdf_with_unstructured(file_path: str) -> List[Document]:
    logger.debug("Loading PDF file with unstructured: %s", file_path)
    elements = partition_pdf(filename=file_path, strategy="ocr_only")
    element_meta = elements[0].to_dict()["metadata"] if elements else {}
    doc_meta = {
        key: element_meta[key]
        for key in [
            "filetype",
            "languages",
            "last_modified",
            "file_directory",
            "filename",
        ]
        if key in element_meta
    }
    doc_meta["source"] = file_path
    content = "\n".join([element.text for element in elements])
    return [Document(page_content=content, metadata=doc_meta)]


def load_pdf_files(
    environment: Environment,
    index_config: IndexConfig,
    file_paths: list[str],
    **kwargs: dict,
):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        environment (Environment): The environment class
        index_config (IndexConfig): The index configuration class.
        file_paths (list[str]): Sequence of paths to load.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """

    logger.info("Loading PDF files")
    documents = []
    load_pdf_from_path = (
        _load_pdf_with_pypdf
        if index_config.pypdf_enabled
        else _load_pdf_with_unstructured
    )
    for file_path in file_paths:
        documents += load_pdf_from_path(file_path)

    logger.debug(f"Loaded {len(documents)} pages from PDF files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=index_config.chunking.chunk_size,
        chunk_overlap=index_config.chunking.overlap_size,
    )

    logger.debug(
        f"Splitting PDF pages into chunks of {index_config.chunking.chunk_size} characters with an overlap of {index_config.chunking.overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)
    docsList = []
    for doc in docs:
        docsList.append(
            {
                str(uuid.uuid4()): {
                    "content": preprocess_pdf_content(doc.page_content),
                    "metadata": doc.metadata,
                }
            }
        )

    logger.info(f"Split {len(documents)} PDF pages into {len(docs)} chunks")

    return docsList
