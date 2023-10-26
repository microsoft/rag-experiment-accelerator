from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level


def load_pdf_files(
    folder_path: str,
    chunk_size: int,
    overlap_size: int,
    glob_pattern: str = "**/[!.]*.{pdf,pdfa,pdfa-1,pdfl}",
):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        folder_path (str): The path to the folder containing the PDF files.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.

    Returns:
        List[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """
    logger.debug(f"Loading PDF files from {folder_path}")
    loader = PyPDFDirectoryLoader(
        path=folder_path,
        glob=glob_pattern,
    )

    documents = loader.load()

    logger.debug(f"Loaded {len(documents)} PDF files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )

    logger.debug(
        f"Splitting PDF files into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)

    logger.debug(f"Split {len(documents)} PDF files into {len(docs)} chunks")

    return docs
