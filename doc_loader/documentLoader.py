import os
import glob
from typing import List, Union
from pdfLoader import load_pdf_files
from htmlLoader import load_html_files

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level

_FORMAT_VERSIONS = {
    "pdf": "**/[!.]*.{pdf,pdfa,pdfa-1,pdfl}",
    "html": "**/[!.]*.{html,htm,xhtml,html5}",
}
_FORMAT_PROCESSORS = {
    "pdf": load_pdf_files,
    "html": load_html_files,
}

def gather_files_by_extension(folder_path: str, extension: str):
    search_pattern = os.path.join(folder_path, f"*.{extension}")
    files_with_extension = glob.glob(search_pattern)
    return files_with_extension

def load_documents(allowed_formats: Union[List[str], str], folder_path: str, chunk_size: int, overlap_size: int):
    if not os.path.exists(folder_path):
        logger.critical(f"Folder {folder_path} does not exist"  )
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    if allowed_formats == "all":
        allowed_formats = _FORMAT_VERSIONS.keys()
    
    logger.debug(f"Loading documents from {folder_path} with allowed formats {', '.join(allowed_formats)}")

    documents = {}

    for format in allowed_formats:
        if format not in _FORMAT_VERSIONS:
            logger.error(f"Format {format} is not supported")
        documents[format] = _FORMAT_PROCESSORS[format](
            folder_path=folder_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            glob_pattern=_FORMAT_VERSIONS[format],
        )
    
    
