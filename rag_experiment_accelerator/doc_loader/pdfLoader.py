from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.doc_loader.documentIntelligenceLoader import azure_document_intelligence_directory_loader
from rag_experiment_accelerator.config.credentials import AzureDocumentIntelligenceCredentials

logger = get_logger(__name__)


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
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "langchain".
        AzureDocumentIntelligenceCredentials (AzureDocumentIntelligenceCredentials): The credentials for Azure Document Intelligence resource.
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "langchain".
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
            documents += azure_document_intelligence_directory_loader(pattern, folder_path, AzureDocumentIntelligenceCredentials.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AzureDocumentIntelligenceCredentials.AZURE_DOCUMENT_INTELLIGENCE_ADMIN_KEY)
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
        f"Splitting PDF pages into chunks of {chunk_size} characters with an"
        f" overlap of {overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)

    logger.info(f"Split {len(documents)} PDF pages into {len(docs)} chunks")

    return docs
