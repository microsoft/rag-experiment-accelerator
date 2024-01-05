import glob
import os

from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def load_structured_files(
    file_format: str,
    language: str,
    loader: BaseLoader,
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str],
    loader_kwargs: dict[any] = None,
):
    """
    Load and process structured files from a given folder path.

    Args:
        file_format (str): The file_format of the documents to be loaded.
        language (str): The language of the documents to be loaded.
        loader (BaseLoader): The document loader object that reads the files.
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["txt", "md"]).
        loader_kwargs (dict[any]): Extra arguments to loader.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.info(f"Loading {file_format} files from {folder_path}")
    matching_files = []
    for pattern in glob_patterns:
        glob_pattern = (  # "." is used for hidden files, "~" is used for Word temporary files
            f"**/[!.~]*.{pattern}"
        )
        full_glob_pattern = os.path.join(folder_path, glob_pattern)
        matching_files += glob.glob(full_glob_pattern, recursive=True)

    logger.debug(f"Found {len(matching_files)} {file_format} files")

    documents = []
    if loader_kwargs is None:
        loader_kwargs = {}

    for file in matching_files:
        document = loader(file, **loader_kwargs).load()
        documents += document

    logger.debug(f"Loaded {len(documents)} {file_format} files")
    if language is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter().from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )

    logger.debug(
        f"Splitting {file_format} files into chunks of {chunk_size} characters"
        f" with an overlap of {overlap_size} characters"
    )

    docs = text_splitter.split_documents(documents)

    logger.info(
        f"Split {len(documents)} {file_format} files into {len(docs)} chunks"
    )

    return docs
