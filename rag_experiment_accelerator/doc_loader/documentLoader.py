import os
from typing import Union
from rag_experiment_accelerator.doc_loader.pdfLoader import load_pdf_files
from rag_experiment_accelerator.doc_loader.htmlLoader import load_html_files
from rag_experiment_accelerator.doc_loader.markdownLoader import load_markdown_files
from rag_experiment_accelerator.doc_loader.textLoader import load_text_files
from rag_experiment_accelerator.doc_loader.jsonLoader import load_json_files
from rag_experiment_accelerator.doc_loader.docxLoader import load_docx_files

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

_FORMAT_VERSIONS = {
    "pdf": ["pdf", "pdfa", "pdfa-1", "pdfl"],
    "html": ["html", "htm", "xhtml", "html5"],
    "markdown": ["md", "markdown"],
    "json": ["json"],
    "text": ["txt", "rtf"],
    "docx": ["docx"],
}
_FORMAT_PROCESSORS = {
    "pdf": load_pdf_files,
    "html": load_html_files,
    "markdown": load_markdown_files,
    "json": load_json_files,
    "text": load_text_files,
    "docx": load_docx_files,
}


def load_documents(
    allowed_formats: Union[list[str], str],
    folder_path: str,
    chunk_size: int,
    overlap_size: int,
):
    """
    Load documents from a folder and process them into chunks.

    Args:
        allowed_formats (Union[list[str], str]): List of formats or 'all' to allow any supported format.
        folder_path (str): Path to the folder containing the documents.
        chunk_size (int): Size of each chunk.
        overlap_size (int): Size of overlap between adjacent chunks.

    Returns:
        list: A list containing processed document chunks.

    Raises:
        FileNotFoundError: When the specified folder does not exist.
    """

    if not os.path.exists(folder_path):
        logger.critical(f"Folder {folder_path} does not exist")
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    if allowed_formats == "all":
        allowed_formats = _FORMAT_VERSIONS.keys()

    logger.debug(
        f"Loading documents from {folder_path} with allowed formats {', '.join(allowed_formats)}"
    )

    documents = {}

    for format in allowed_formats:
        if format not in _FORMAT_VERSIONS:
            logger.error(f"Format {format} is not supported")
            continue
        documents[format] = _FORMAT_PROCESSORS[format](
            folder_path=folder_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            glob_patterns=_FORMAT_VERSIONS[format],
        )

    all_documents = sum(documents.values(), [])

    logger.info(f"Loaded {len(all_documents)} chunks from {folder_path}")

    return all_documents
