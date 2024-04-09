import uuid
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def preprocess_pdf_content(content: str):
    """
    Preprocess the content extracted from a PDF file.
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
    environment: Environment,
    file_paths: list[str],
    chunk_size: int,
    overlap_size: int,
    **kwargs: dict,
):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """

    logger.info("Loading PDF files")
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path=file_path)
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
