from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def get_supported_formats() -> list[str]:
    """
    Get supported formats for the Azure Document Intelligence loader.

    Returns:
        list[str]: List of formats.
    """

    return ["pdf", "html", "markdown", "json", "text", "docx"]


def load_with_azure_document_intelligence(
    environment: Environment,
    file_paths: list[str],
    chunk_size: int,
    overlap_size: int,
) -> list[Document]:
    """
    Load pdf files from a folder using Azure Document Intelligence.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (int): Unused.
        overlap_size (int): Unused.

    Returns:
        list[Document]: A list of Document objects.
    """
    documents: list[Document] = []
    for file_path in file_paths:
        try:
            documents += AzureAIDocumentIntelligenceLoader(
                file_path=file_path,
                api_key=environment.azure_document_intelligence_admin_key,
                api_endpoint=environment.azure_document_intelligence_endpoint,
                api_model="prebuilt-read",
            ).load()
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    return documents
