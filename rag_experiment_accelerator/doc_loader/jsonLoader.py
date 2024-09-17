from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.doc_loader.customJsonLoader import (
    CustomJSONLoader,
)
from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_json_files(
    environment: Environment,
    index_config: IndexConfig,
    file_paths: list[str],
    **kwargs: dict,
):
    """
    Load and process Json files from a given folder path.

    Args:
        environment (Environment): The environment class.
        index_config (IndexConfig): The index configuration class.
        file_paths (list[str]): Sequence of paths to load.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading json files")

    keys_to_load = ["content", "title"]
    return load_structured_files(
        file_format="JSON",
        language=None,
        loader=CustomJSONLoader,
        file_paths=file_paths,
        chunk_size=index_config.chunking.chunk_size,
        overlap_size=index_config.chunking.overlap_size,
        loader_kwargs={
            "keys_to_load": keys_to_load,
        },
    )
