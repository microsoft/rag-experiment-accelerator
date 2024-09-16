from langchain_community.document_loaders import UnstructuredMarkdownLoader

from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_markdown_files(
    environment: Environment,
    file_paths: list[str],
    chunk_size: str,
    overlap_size: str,
    **kwargs: dict,
):
    """
    Load and process Markdown files from a given folder path.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading markdown files")

    return load_structured_files(
        file_format="MARKDOWN",
        language="markdown",
        loader=UnstructuredMarkdownLoader,
        file_paths=file_paths,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )
