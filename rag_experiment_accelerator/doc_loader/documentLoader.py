from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.doc_loader.docxLoader import load_docx_files
from rag_experiment_accelerator.doc_loader.htmlLoader import load_html_files
from rag_experiment_accelerator.doc_loader.jsonLoader import load_json_files
from rag_experiment_accelerator.doc_loader.markdownLoader import (
    load_markdown_files,
)
from rag_experiment_accelerator.doc_loader.pdfLoader import load_pdf_files
from rag_experiment_accelerator.doc_loader.textLoader import load_text_files
from rag_experiment_accelerator.doc_loader.documentIntelligenceLoader import (
    is_supported_by_document_intelligence,
    load_with_azure_document_intelligence,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.chunking_config import ChunkingStrategy

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


def determine_processor(chunking_strategy: ChunkingStrategy, format: str) -> callable:
    """
    Determine and return document processor based on chunking strategy and format.
    """
    if (
        chunking_strategy == ChunkingStrategy.AZURE_DOCUMENT_INTELLIGENCE
        and is_supported_by_document_intelligence(format)
    ):
        return load_with_azure_document_intelligence
    else:
        return _FORMAT_PROCESSORS[format]


def load_documents(
    environment: Environment,
    index_config: IndexConfig,
    allowed_formats: list[str],
    file_paths: list[str],
):
    """
    Load documents from a folder and process them into chunks.

    Args:
        environment (Environment): The environment class
        index_config (IndexConfig): The index configuration class.
        allowed_formats (list[str]]): List of formats, ['*'] - to allow any supported format.
        folder_path (str): Path to the folder containing the documents.

    Returns:
        list: A list of dictionaries containing the processed chunks.

    Raises:
        FileNotFoundError: When the specified folder does not exist.
    """

    # ['*'] - to allow any supported format.
    if "*" in allowed_formats:
        allowed_formats = _FORMAT_VERSIONS.keys()

    logger.debug(f"Loading documents with allowed formats {', '.join(allowed_formats)}")

    documents = {}

    for format in allowed_formats:
        if format not in _FORMAT_VERSIONS:
            logger.error(f"Format {format} is not supported")
            continue
        matching_files = [
            path
            for path in file_paths
            if any(path.endswith(pattern) for pattern in _FORMAT_VERSIONS[format])
        ]

        processor = determine_processor(
            chunking_strategy=index_config.chunking.chunking_strategy, format=format
        )
        documents[format] = processor(
            environment=environment,
            index_config=index_config,
            file_paths=matching_files,
        )

    all_documents = []
    for inner_dict in documents.keys():
        for value in documents[inner_dict]:
            all_documents.append(value)

    logger.info(f"Loaded {len(all_documents)} chunks")
    return all_documents
