from typing import Iterable

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def load_with_azure_document_intelligence(
    environment: Environment,
    file_paths: Iterable[str],
    chunk_size: int,
    overlap_size: int,
) -> Iterable[Document]:
    """
    Load pdf files from a folder using Azure Document Intelligence.

    Args:
        file_paths (Iterable[str]): Sequence of paths to load.
        environment (Environment): The environment class
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
            pass

    return documents
