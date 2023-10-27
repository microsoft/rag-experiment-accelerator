from typing import List
from doc_loader.structuredLoader import load_structured_files
from langchain.document_loaders import UnstructuredMarkdownLoader
from utils.logging import get_logger

logger = get_logger(__name__)


def load_markdown_files(
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: List[str] = ["html", "htm", "xhtml", "html5"],
):
    """
    Load and process Markdown files from a given folder path.

    Args:
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (List[str]): List of file extensions to consider (e.g., ["md", "markdown", ...]).

    Returns:
        List[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading markdown files")

    return load_structured_files(
        language="markdown",
        loader=UnstructuredMarkdownLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
    )
