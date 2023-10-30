import os
import glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.base import BaseLoader
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def load_structured_files(
    language: str,
    loader: BaseLoader,
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: List[str],
):
    """
    Load and process structured files from a given folder path.

    Args:
        language (str): The language of the documents to be loaded.
        loader (BaseLoader): The document loader object that reads the files.
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (List[str]): List of file extensions to consider (e.g., ["txt", "md"]).

    Returns:
        List[Document]: A list of processed and split document chunks.
    """

    logger.info(f"Loading {language.upper()} files from {folder_path}")
    matching_files = []
    for pattern in glob_patterns:
        glob_pattern = f"**/[!.]*.{pattern}"
        full_glob_pattern = os.path.join(folder_path, glob_pattern)
        matching_files += glob.glob(full_glob_pattern, recursive=True)

    logger.debug(f"Found {len(matching_files)} {language.upper()} files")

    documents = []
    for file in matching_files:
        document = loader(file).load()
        documents += document

    logger.debug(f"Loaded {len(documents)} {language.upper()} files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    ).from_language(language=language)

    logger.debug(
        f"Splitting {language.upper()} files into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )

    docs = text_splitter.split_documents(documents)

    logger.info(
        f"Split {len(documents)} {language.upper()} files into {len(docs)} chunks"
    )

    return docs
