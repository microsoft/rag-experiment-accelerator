import uuid

from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def load_structured_files(
    file_format: str,
    language: str,
    loader: BaseLoader,
    file_paths: list[str],
    chunk_size: str,
    overlap_size: str,
    loader_kwargs: dict[any] = None,
):
    """
    Load and process structured files.

    Args:
        chunking_strategy (str): The chunking strategy to use between "azure-document-intelligence" and "basic".
        file_format (str): The file_format of the documents to be loaded.
        language (str): The language of the documents to be loaded.
        loader (BaseLoader): The document loader object that reads the files.
        file_paths (str): The paths to the files to load.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["txt", "md"]).
        loader_kwargs (dict[any]): Extra arguments to loader.

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.info(f"Loading {file_format} files")

    documents = []
    if loader_kwargs is None:
        loader_kwargs = {}

    for file in file_paths:
        documents += loader(file, **loader_kwargs).load()

    logger.debug(f"Loaded {len(documents)} {file_format} files")
    if language is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter().from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )

    logger.debug(
        f"Splitting {file_format} files into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )

    docs = text_splitter.split_documents(documents)
    docsList = []
    for doc in docs:
        docsList.append(
            {str(uuid.uuid4()): {"content": doc.page_content, "metadata": doc.metadata}}
        )

    logger.info(f"Split {len(documents)} {file_format} files into {len(docs)} chunks")

    return docsList
