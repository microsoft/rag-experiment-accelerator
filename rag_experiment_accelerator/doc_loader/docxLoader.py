from langchain_community.document_loaders import Docx2txtLoader

from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_docx_files(
    environment: Environment,
    file_paths: list[str],
    chunk_size: str,
    overlap_size: str,
    **kwargs: dict,
):
    """
    Load and process docx files from a given folder path.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.
        **kwargs (dict): Unused.


    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading docx files")

    return load_structured_files(
        file_format="DOCX",
        language=None,
        loader=Docx2txtLoader,
        file_paths=file_paths,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )
