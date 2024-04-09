from langchain_community.document_loaders import BSHTMLLoader

from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_html_files(
    environment: Environment,
    file_paths: list[str],
    chunk_size: str,
    overlap_size: str,
    **kwargs: dict,
):
    """
    Load and process HTML files from a given folder path.

    Args:
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "basic".
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["html", "htm", ...]).
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading html files")

    return load_structured_files(
        file_format="HTML",
        language="html",
        loader=BSHTMLLoader,
        file_paths=file_paths,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        loader_kwargs={"open_encoding": "utf-8"},
    )
