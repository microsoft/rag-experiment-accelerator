from langchain.document_loaders import UnstructuredMarkdownLoader

from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.credentials import AzureDocumentIntelligenceCredentials

logger = get_logger(__name__)


def load_markdown_files(
    chunking_strategy,
    AzureDocumentIntelligenceCredentials: AzureDocumentIntelligenceCredentials,
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
):
    """
    Load and process Markdown files from a given folder path.

    Args:
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "langchain".
        AzureDocumentIntelligenceCredentials (AzureDocumentIntelligenceCredentials): The credentials for Azure Document Intelligence resource.
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["md", "markdown", ...]).

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading markdown files")

    return load_structured_files(
        chunking_strategy,
        AzureDocumentIntelligenceCredentials,
        file_format="MARKDOWN",
        language="markdown",
        loader=UnstructuredMarkdownLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
    )
