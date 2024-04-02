from langchain_community.document_loaders import BSHTMLLoader
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import spacy

from rag_experiment_accelerator.doc_loader.utils.semantic_chunking import (
    get_semantic_similarity
)
from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.config import ChunkingStrategy, Config

logger = get_logger(__name__)


def load_html_files(
    environment: Environment,
    config: Config,
    file_paths: list[str],
    chunk_size: str,
    overlap_size: str,
):
    """
    Load and process HTML files from a given folder path.

    Args:
        environment (Environment): The environment class
        config (Config): The configuration class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading html files")

    if config.CHUNKING_STRATEGY == ChunkingStrategy.SEMANTIC:
        return _load_html_files_semantic(
            file_paths=file_paths,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            similarity_score=config.SEMANTIC_SIMILARITY_THRESHOLD
        )

    else:
        return load_structured_files(
            file_format="HTML",
            language="html",
            loader=BSHTMLLoader,
            file_paths=file_paths,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            loader_kwargs={"open_encoding": "utf-8"},
        )


def _load_html_files_semantic(
    file_paths: list[str],
    chunk_size: int = 0,
    overlap_size: int = 0,
    similarity_score: float = 0.95
):
    """Internal method to load and process HTML files from a given folder path using semantic chunking.

    Args:
        file_paths (str): List of files to be parsed.
        chunk_size (int, optional): The soft maximum number of tokens in a chunk. Defaults to 0.
        overlap_size (int, optional): The maximum number of tokens to overlap between chunks. Defaults to 0.
        similarity_score (float, optional): The similarity score to use for semantic chunking. Defaults to 0.95.
    """

    logger.info("Loading html files semantically")

    nlp = spacy.load("en_core_web_md")
    unique_dict = {}
    all_chunks = {}

    for filename in file_paths:
        elements = partition_html(
            filename=filename
        )

        # Currently using titles as the semantic segment definition
        chunks = chunk_by_title(
            elements=elements,
            new_after_n_chars=chunk_size,
            overlap=overlap_size
        )

        embeddings_dict = {}
        for chunk in chunks:
            # Skip chunks that are a single word
            if len(chunk.text.split()) > 2:
                if chunk.id not in unique_dict:
                    # Create a Document out of the chunk)
                    unique_dict[chunk.id] = {"content": chunk.text, "metadata": chunk.metadata.to_dict()}
                # TODO: Allow embedding model to be configurable. Currently uses small spacy model for speed
                doc = nlp(chunk.text)
                embeddings_dict[chunk.id] = doc.vector
        high_similarity, low_similarity = get_semantic_similarity(embeddings_dict, unique_dict, similarity_score)
        all_chunks.update(high_similarity)
        all_chunks.update(low_similarity)
    return (all_chunks)
