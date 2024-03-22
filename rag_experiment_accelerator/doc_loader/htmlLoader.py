from glob import glob
import os
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document
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
from rag_experiment_accelerator.config.credentials import (
    AzureDocumentIntelligenceCredentials,
)

logger = get_logger(__name__)


def load_html_files(
    chunking_strategy,
    AzureDocumentIntelligenceCredentials: AzureDocumentIntelligenceCredentials,
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
    similarity_score: float = 0.95
):
    """
    Load and process HTML files from a given folder path.

    Args:
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence", "semantic", and "basic".
        AzureDocumentIntelligenceCredentials (AzureDocumentIntelligenceCredentials): The credentials for Azure Document Intelligence resource.
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["html", "htm", ...]).
        similarity_score (float): The similarity score to use for semantic chunking. Defaults to 0.95.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading html files")

    if chunking_strategy != "semantic":
        return load_structured_files(
            chunking_strategy,
            AzureDocumentIntelligenceCredentials,
            file_format="HTML",
            language="html",
            loader=BSHTMLLoader,
            folder_path=folder_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            glob_patterns=glob_patterns,
            loader_kwargs={"open_encoding": "utf-8"},
        )
    else:
        return _load_html_files_semantic(
            folder_path=folder_path,
            glob_patterns=glob_patterns,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            similarity_score=similarity_score
        )


def _load_html_files_semantic(    
    folder_path: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
    chunk_size: int = 0,
    overlap_size: int = 0,
    similarity_score: float = 0.95
):
    """Internal method to load and process HTML files from a given folder path using semantic chunking.

    Args:
        folder_path (str): The path of the folder where files are located.
        glob_patterns (list[str], optional): List of file extensions to consider. Defaults to ["html", "htm", "xhtml", "html5"].
        chunk_size (int, optional): The soft maximum number of tokens in a chunk. Defaults to 0.
        overlap_size (int, optional): The maximum number of tokens to overlap between chunks. Defaults to 0.
        similarity_score (float, optional): The similarity score to use for semantic chunking. Defaults to 0.95.
    """

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
        ## TODO: confirm this returns metadata
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
                    unique_dict[chunk.id] = chunk.text
                # TODO: Consider using a different embedding model, or configurable model
                doc = nlp(chunk.text)
                embeddings_dict[chunk.id] = doc.vector
        high_similarity, low_similarity = get_semantic_similarity(embeddings_dict, unique_dict, similarity_score)
        all_chunks.update(high_similarity) 
        all_chunks.update(low_similarity)
    return(all_chunks)


