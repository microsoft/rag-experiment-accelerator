from langchain_community.document_loaders import TextLoader

from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_text_files(
    environment: Environment,
    index_config: IndexConfig,
    file_paths: list[str],
    **kwargs: dict,
):
    """
    Load and process text files from a given folder path.

    Args:
        environment (Environment): The environment class.
        index_config (IndexConfig): The index configuration class.
        file_paths (list[str]): Sequence of paths to load.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading text files")

    return load_structured_files(
        file_format="TEXT",
        language=None,
        loader=TextLoader,
        file_paths=file_paths,
        chunk_size=index_config.chunking.chunk_size,
        overlap_size=index_config.chunking.overlap_size,
    )
