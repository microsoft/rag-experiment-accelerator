from typing import Iterable
import uuid

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_pdf_files(
    environment: Environment,
    file_paths: Iterable[str],
    chunk_size: int,
    overlap_size: int,
):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        environment (Environment): The environment class
        file_paths (Iterable[str]): Sequence of paths to load.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """

    logger.info("Loading PDF files")
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path=file_path)
        documents += loader.load()

    logger.debug(f"Loaded {len(documents)} pages from PDF files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )

    logger.debug(
        f"Splitting PDF pages into chunks of {chunk_size} characters with an"
        f" overlap of {overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)
    docsList = []
    for doc in docs:
        docsList.append(dict({str(uuid.uuid4()): doc.page_content}))

    logger.info(f"Split {len(documents)} PDF pages into {len(docs)} chunks")

    return docsList
