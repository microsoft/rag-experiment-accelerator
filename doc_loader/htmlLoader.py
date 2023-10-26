import os
import glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level


def load_html_files(
        folder_path: str, 
        chunk_size: str,
        overlap_size: str,
        glob_patterns: List[str] = ["html", "htm", "xhtml", "html5"],
    ):
    
    logger.debug(f"Loading HTML files from {folder_path}")
    matching_files = []
    for pattern in glob_patterns:
        glob_pattern = f"**/[!.]*.{pattern}"
        full_glob_pattern = os.path.join(folder_path, glob_pattern)
        matching_files += glob.glob(full_glob_pattern, recursive=True)

    logger.debug(f"Found {len(matching_files)} HTML files")

    documents = []
    for file in matching_files:
        document = BSHTMLLoader(
            file
        ).load()
        documents += document

    logger.debug(f"Loaded {len(documents)} HTML files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    ).from_language("html")

    logger.debug(
        f"Splitting HTML files into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )

    docs = text_splitter.split_documents(documents)

    logger.debug(f"Split {len(documents)} HTML files into {len(docs)} chunks")

    return docs