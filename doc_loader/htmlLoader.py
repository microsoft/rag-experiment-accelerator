import os
from pathlib import Path
from typing import List, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import BSHTMLLoader, BaseLoader
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level


class BSHTMLDirectoryLoader(BaseLoader):
    def __init__(
            self,
            path: str,
            glob: str = "**/[!.]*.html",
            open_encoding: Union[str, None] = None,
            silent_errors: bool = False,
            load_hidden: bool = False,
            recursive: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = BSHTMLLoader(str(i))
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            print(doc.metadata)
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs




def load_html_files(
        folder_path: str, 
        chunk_size: str,
        overlap_size: str,
        glob_pattern: str = "**/[!.]*.{html,htm,xhtml,html5}",
    ):
    
    logger.debug(f"Loading HTML files from {folder_path}")
    loader = BSHTMLDirectoryLoader(
        path=folder_path,
        glob=glob_pattern,
    )

    documents = loader.load()

    logger.debug(f"Loaded {len(documents)} HTML files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )

    logger.debug(
        f"Splitting HTML files into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )
    docs = text_splitter.split_documents(documents)

    logger.debug(f"Split {len(documents)} HTML files into {len(docs)} chunks")

    return docs