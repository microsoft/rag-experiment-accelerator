import os

from rag_experiment_accelerator.utils.logging import get_logger
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from pathlib import Path

logger = get_logger(__name__)


def azure_document_intelligence_loader(pattern, folder_path, endpoint, key):
    """
    Load pdf files from a folder using Azure Document Intelligence.

    Args:
        pattern (str): The file extension to look for.
        folder_path (str): The path to the folder containing the files.
        endpoint (str): The Azure Document Intelligence endpoint.
        key (str): The Azure Document Intelligence key.

    Returns:
        list[Document]: A list of Document objects.
    """

    glob = f"**/[!.]*.{pattern}"
    p = Path(folder_path)
    documents = []
    items = p.glob(glob)
    for i in items:
        if i.is_file():
            try:
                print("file name ", i)
                loader = AzureAIDocumentIntelligenceLoader(file_path=i, api_key = key, api_endpoint = endpoint, api_model="prebuilt-read")
                documents += loader.load()
            except Exception as e:
                logger.warning(f"Failed to load {pattern} file {i}: {e}")
                continue

    return documents
