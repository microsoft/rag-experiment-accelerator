from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.doc_loader.documentIntelligenceLoader import (
    azure_document_intelligence_directory_loader,
)
from rag_experiment_accelerator.config.credentials import (
    AzureDocumentIntelligenceCredentials,
)

import uuid
import re

logger = get_logger(__name__)


def preprocess_pdf_content(content: str):
    """
    Preprocesses the content extracted from a PDF file.
    This function performs the following preprocessing steps on the input content:
    1. Replaces multiple consecutive newline characters ('\\n') with a single newline character.
    2. Removes all remaining newline characters.
    3. Removes Unicode escape sequences in the format '\\uXXXX' where X is a hexadecimal digit.
    4. Converts the content to lowercase.
    Args:
        content (str): The content extracted from the PDF file.
    Returns:
        str: The preprocessed content.
    Example:
        content = "Hello\\n\\nWorld\\n\\u1234 OpenAI"
        preprocessed_content = preprocess_pdf_content(content)
        print(preprocessed_content)
        # Output: "hello world openai"
    """

    content = re.sub(r"\n{2,}", "\n", content)
    content = re.sub(r"\n{1,}", "", content)
    content = re.sub(r"\\u[0-9a-fA-F]{4}", "", content)
    content = content.lower()

    return content


def load_pdf_files(
    chunking_strategy,
    AzureDocumentIntelligenceCredentials: AzureDocumentIntelligenceCredentials,
    folder_path: str,
    chunk_size: int,
    overlap_size: int,
    glob_patterns: list[str] = ["pdf", "pdfa", "pdfa-1", "pdfl"],
):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "basic".
        AzureDocumentIntelligenceCredentials (AzureDocumentIntelligenceCredentials): The credentials for Azure Document Intelligence resource.
        folder_path (str): The path to the folder containing the PDF files.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """

    logger.info(f"Loading PDF files from {folder_path}")
    documents = []
    for pattern in glob_patterns:
        if chunking_strategy == "azure-document-intelligence":
            documents += azure_document_intelligence_directory_loader(
                pattern,
                folder_path,
                AzureDocumentIntelligenceCredentials.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                AzureDocumentIntelligenceCredentials.AZURE_DOCUMENT_INTELLIGENCE_ADMIN_KEY,
            )
        else:
            # using langchain
            loader = PyPDFDirectoryLoader(
                path=folder_path,
                glob=f"**/[!.]*.{pattern}",
                recursive=True,
            )
            documents += loader.load()

    logger.debug(f"Loaded {len(documents)} pages from PDF files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )

    logger.debug(
        f"Splitting PDF pages into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)
    docsList = []
    for doc in docs:
        docsList.append(
            {
                str(uuid.uuid4()): {
                    "content": preprocess_pdf_content(doc.page_content),
                    "metadata": doc.metadata,
                }
            }
        )

    logger.info(f"Split {len(documents)} PDF pages into {len(docs)} chunks")

    return docsList
