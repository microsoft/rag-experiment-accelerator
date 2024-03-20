from glob import glob
import os
import json
import hashlib
import random
import string
import uuid
from langchain.document_loaders import BSHTMLLoader
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json
import spacy

from rag_experiment_accelerator.doc_loader.utils.semantic_chunking import (
    get_semantic_similarity
)
from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def load_html_files(
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
):
    """
    Load and process HTML files from a given folder path.

    Args:
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["html", "htm", ...]).

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading html files")

    return load_structured_files(
        file_format="HTML",
        language="html",
        loader=BSHTMLLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
        loader_kwargs={"open_encoding": "utf-8"},
    )


def load_html_files_semantic(    
    folder_path: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
    chunk_size: int = 0,
    overlap_size: int = 0,
):
    nlp = spacy.load("en_core_web_md")
    unique_dict = {}
    all_chunks = {}
    matching_files = []
    for pattern in glob_patterns:
        # "." is used for hidden files, "~" is used for Word temporary files
        glob_pattern = f"**/[!.~]*.{pattern}"
        full_glob_pattern = os.path.join(folder_path, glob_pattern)
        matching_files += glob(full_glob_pattern, recursive=True)

    for filename in matching_files:
        elements = partition_html(
            filename=filename
        )

        # TODO: Why are titles separated?
        chunks = chunk_by_title(
            elements=elements,
            new_after_n_chars=chunk_size,
            overlap=overlap_size
        )

        embeddings_dict = {}
        # TODO: If it's a single word - don't chunk it. Toss it.
        for chunk in chunks:
            if chunk.id not in unique_dict:
                unique_dict[chunk.id] = chunk.text
            doc = nlp(chunk.text)
            # TODO: Consider using a different embedding model, or configurable model
            embeddings_dict[chunk.id] = doc.vector
        # TODO: Parameterize the similarity score
        high_similarity, low_similarity = get_semantic_similarity(embeddings_dict, unique_dict, 0.95)
        all_chunks.update(high_similarity)
        all_chunks.update(low_similarity)
    return(all_chunks)


