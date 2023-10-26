import os
from typing import List
from doc_loader.structuredLoader import load_structured_files
from langchain.document_loaders import UnstructuredMarkdownLoader
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level


def load_markdown_files(
        folder_path: str, 
        chunk_size: str,
        overlap_size: str,
        glob_patterns: List[str] = ["html", "htm", "xhtml", "html5"],
    ):
    
    return load_structured_files(
        language="markdown",
        loader=UnstructuredMarkdownLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
    )