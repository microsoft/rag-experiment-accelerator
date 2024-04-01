import uuid
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def is_supported_by_document_intelligence(format: str) -> bool:
    """
    Returns whether a format is supported by Azure Document Intelligence or not.

    Returns:
        bool: True if the format is supported, False otherwise.
    """

    return format.lower() in [
        "pdf",
        "jpeg",
        "jpg",
        "png",
        "bmp",
        "heif",
        "tiff",
        "docx",
        "xlsx",
        "pptx",
        "html",
    ]


def load_with_azure_document_intelligence(
    environment: Environment,
    file_paths: list[str],
    azure_document_intelligence_model: str,
    **kwargs: dict,
) -> list[Document]:
    """
    Load pdf files from a folder using Azure Document Intelligence.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        azure_document_intelligence_model (str): The model to use for Azure Document Intelligence.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of Document objects.
    """
    documents = []
    logger.info(f"Using model {azure_document_intelligence_model}")
    for file_path in file_paths:
        try:
            documents.append({
                "content": AzureAIDocumentIntelligenceLoader(
                    file_path=file_path,
                    api_key=environment.azure_document_intelligence_admin_key,
                    api_endpoint=environment.azure_document_intelligence_endpoint,
                    api_model=azure_document_intelligence_model,
                ).load()[0].page_content,
                "metadata": {
                    "source": file_path,
                    "page": 0
                }})
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    return [{str(uuid.uuid4()): doc} for doc in documents]
