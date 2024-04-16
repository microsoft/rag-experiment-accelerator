import json
from pathlib import Path
from typing import Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

# Replaces langchain.document_loaders.JSONLoader to not use jq for windows compatibility
# Note: Does not currently support jsonl, which is what the seq_num metadata field tracks


class CustomJSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        keys_to_load: list[str] = ["content", "title"],
        strict_keys: bool = True,
    ):
        self.file_path = Path(file_path).resolve()
        self._keys_to_load = keys_to_load
        self._strict_keys = strict_keys

    def _load_schema_from_dict(self, data: dict) -> str:
        if self._keys_to_load is None:
            return data
        else:
            return_dict = {}
            for k in self._keys_to_load:
                value = data.get(k)
                if value is None and self._strict_keys:
                    raise ValueError(
                        f"JSON file at path {self.file_path} must contain the field '{k}'"
                    )
                return_dict[k] = value
        return return_dict

    def load(self) -> list[Document]:
        """Load and return documents from the JSON file."""
        docs: list[Document] = []
        # Load JSON file
        with self.file_path.open(encoding="utf-8") as f:
            data = json.load(f)
            page_content = []

            if not isinstance(data, list):
                raise ValueError(
                    f"JSON file at path: {self.file_path} must be a list of object and expects each object to contain the fields {self._keys_to_load}"
                )
            else:
                for entry in data:
                    data_dict = self._load_schema_from_dict(entry)
                    page_content.append(data_dict)

            metadata = {
                "source": str(self.file_path),
            }

            docs.append(Document(page_content=str(page_content), metadata=metadata))
        return docs
