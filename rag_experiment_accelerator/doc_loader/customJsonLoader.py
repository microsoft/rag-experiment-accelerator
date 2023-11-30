import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

# Replaces langchain.document_loaders.JSONLoader to not use jq for windows compatibility
class JSoNLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
        # json_lines: bool = False,
        ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        # self._json_lines = json_lines

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs: List[Document] = []
        # Load JSON file
        with self.file_path.open(encoding="utf-8") as f:
            data = json.load(f)
            page_content = []

            for entry in data:
                page_content.append({
                    'content': entry['content'],
                    'title': entry['title']
                })

            metadata = {
                'source': self.file_path,
            }
            docs.append(Document(page_content=str(page_content), metadata=metadata))
        return docs
