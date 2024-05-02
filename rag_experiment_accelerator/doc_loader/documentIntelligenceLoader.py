from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import re
import os
import uuid
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_experiment_accelerator.config.environment import Environment
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document
from pathlib import Path

from langchain_community.document_loaders.base import BaseLoader
from typing import List, Iterator
from rag_experiment_accelerator.utils.logging import get_logger
from azure.ai.documentintelligence.models import DocumentParagraph

logger = get_logger(__name__)


def is_supported_by_document_intelligence(format: str) -> bool:
    """
    Returns whether a format is supported by Azure Document Intelligence or not.

    Returns:
        bool: True if the format is supported, False otherwise.
    """

    return format.lower() in [
        "pdf",
        "jpeg",
        "jpg",
        "png",
        "bmp",
        "heif",
        "tiff",
        "docx",
        "xlsx",
        "pptx",
        "html",
    ]


def load_with_azure_document_intelligence(
    environment: Environment,
    file_paths: list[str],
    chunk_size: int,
    overlap_size: int,
    azure_document_intelligence_model: str,
    **kwargs: dict,
) -> list[Document]:
    """
    Load pdf files from a folder using Azure Document Intelligence.

    Args:
        environment (Environment): The environment class
        file_paths (list[str]): Sequence of paths to load.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.
        azure_document_intelligence_model (str): The model to use for Azure Document Intelligence.
        **kwargs (dict): Unused.

    Returns:
        list[Document]: A list of Document objects.
    """
    documents = []
    logger.info(f"Using model {azure_document_intelligence_model}")
    for file_path in file_paths:
        try:
            loader = DocumentIntelligenceLoader(
                file_path,
                environment.azure_document_intelligence_endpoint,
                environment.azure_document_intelligence_admin_key,
                azure_document_intelligence_model,
                glob_patterns=["*"],
                excluded_paragraph_roles=[
                    "pageHeader",
                    "pageFooter",
                    "footnote",
                    "pageNumber",
                ],
            )
            documents += loader.load()
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    logger.debug(f"Loaded {len(documents)} documents using Azure Document Intelligence")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n"],
    )

    logger.debug(
        f"Splitting extracted documents into chunks of {chunk_size} characters with an overlap of {overlap_size} characters"
    )

    docs = text_splitter.split_documents(documents)

    return [{str(uuid.uuid4()): doc.__dict__} for doc in docs]


class DocumentIntelligenceLoader(BaseLoader):
    """
    Analyzes and loads documents and directories using Azure Document Intelligence.
    """

    def __init__(
        self,
        path: str,
        endpoint: str,
        key: str,
        api_model: str,
        glob_patterns: List[str] = None,
        split_documents_by_page=False,
        excluded_paragraph_roles=[],
        patterns_to_remove: List[str] = [],
    ):
        """
        Initializes an instance of the DocumentIntelligenceLoader class.

        Parameters:
            path: path of the document or directory to load from, when a directory path is provided a glob_pattern has to be provided as well
            end_point: Azure Document Intelligence endpoint
            key: Azure Document Intelligence key
            api_model (str): The model to use for Azure Document Intelligence.
            glob_patterns: when the given path is a directory, glob_patterns is used to match the files that should be loaded
            split_documents_by_page: if True, each page in the document will be loaded into separate LangChain document, otherwise (default) the entire document will be loaded into a single LangChain document
            excluded_paragraph_roles: a list of paragraph roles to exclude. The full list of paragraph roles can be viewed here: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout?view=doc-intel-4.0.0#paragraph-roles
            patterns_to_remove: a list of specific regex patterns to be removed from the extracted text
        """
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        self.path = path
        self.endpoint = endpoint
        self.key = key
        self.api_model = api_model
        self.patterns_to_remove = patterns_to_remove
        self.glob_patterns = glob_patterns
        self.split_documents_by_page = split_documents_by_page
        self.excluded_paragraph_roles = excluded_paragraph_roles

    def load(self) -> List[Document]:
        documents = []
        file_paths = self._get_file_paths()

        with ExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor())

            futures = {
                executor.submit(self._analyze_document, file_path)
                for file_path in file_paths
            }

            for future in as_completed(futures):
                try:
                    documents += future.result()
                except Exception as exc:
                    logger.error(f"Processing document generated an exception: {exc}")

            return documents

    def lazy_load(self) -> Iterator[Document]:
        file_paths = self._get_file_paths()
        for file_path in file_paths:
            yield self._analyze_document(file_path)

    def _get_file_paths(self):
        if not os.path.isdir(self.path):
            return [self.path]

        directory = Path(self.path)
        file_paths = []
        for pattern in self.glob_patterns:
            file_paths += list(directory.rglob(f"*.{pattern}"))

        return [str(path) for path in file_paths]

    def _analyze_document(self, file_path: str):
        if not self.api_model == "prebuilt-layout":
            return self._load_with_langchain(file_path, self.api_model)

        documents = []
        try:
            result = self._call_document_intelligence(file_path)

            if result.tables:
                paragraphs = self._substitute_table_paragraphs(
                    result.paragraphs, result.tables
                )
            else:
                paragraphs = result.paragraphs

            relevant_paragraphs = []
            for paragraph in paragraphs:
                if "role" in paragraph.keys():
                    if paragraph["role"] not in self.excluded_paragraph_roles:
                        relevant_paragraphs.append(paragraph)
                else:
                    relevant_paragraphs.append(paragraph)

            if self.split_documents_by_page:
                paragraphs_by_page = self._split_paragraphs_by_page(relevant_paragraphs)
                for page_number, page_paragraphs in paragraphs_by_page.items():
                    documents.append(
                        self._convert_to_langchain_document(
                            page_paragraphs, file_path, page_number
                        )
                    )
            else:
                documents.append(
                    self._convert_to_langchain_document(
                        relevant_paragraphs, file_path, 1
                    )
                )

            return documents
        except Exception as exc:
            logger.warning(
                f"Failed to load {file_path} with Azure Document Intelligence using the 'prebuilt-layout' model: {exc}. Attempting to load using the simpler 'prebuilt-read' model..."
            )
            return self._load_with_langchain(file_path, "prebuilt-read")

    def _call_document_intelligence(self, file_path):
        with open(file_path, "rb") as file:
            content = file.read()
            poller = self.client.begin_analyze_document(
                "prebuilt-layout",
                content,
                content_type="application/octet-stream",
                output_content_format="markdown",
            )

        result = poller.result()
        return result

    def _clean_content(self, content: str):
        # Remove AI doc intelligence traces.
        pattern = re.compile(r":selected:|:unselected:")
        content = pattern.sub("", content)
        # Remove specific regex patterns.
        for regex_pattern in self.patterns_to_remove:
            pattern = re.compile(regex_pattern)
            content = pattern.sub("", content)

        return content

    def _convert_to_langchain_document(self, paragraphs, file_path, page_number):
        content = "\n\n".join([paragraph.content for paragraph in paragraphs])
        clean_content = self._clean_content(content)
        return Document(
            page_content=clean_content,
            metadata={"source": file_path, "page": page_number - 1},
        )

    def _is_intersecting_regions(self, bounding_region1, bounding_region2):
        """
        Returns whether two bounding regions intersect or not
        """
        for region1 in bounding_region1:
            for region2 in bounding_region2:
                if region1["pageNumber"] == region2[
                    "pageNumber"
                ] and self._is_intersecting_polygons(region1.polygon, region2.polygon):
                    return True
        return False

    def _is_intersecting_polygons(self, polygon1, polygon2):
        """
        Returns whether two polygons intersect or not
        """
        x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1 = polygon1
        x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2 = polygon2

        # Check for overlap along the x-axis
        if max(x1_1, x2_1, x3_1, x4_1) < min(x1_2, x2_2, x3_2, x4_2) or min(
            x1_1, x2_1, x3_1, x4_1
        ) > max(x1_2, x2_2, x3_2, x4_2):
            return False

        # Check for overlap along the y-axis
        if max(y1_1, y2_1, y3_1, y4_1) < min(y1_2, y2_2, y3_2, y4_2) or min(
            y1_1, y2_1, y3_1, y4_1
        ) > max(y1_2, y2_2, y3_2, y4_2):
            return False

        # If the boxes overlap along both axes, they intersect
        return True

    def _assign_tables_to_paragraphs(self, paragraphs, tables):
        """
        Returns a list that maps paragraph indexes to their tables indexes.
        Indexes in the returned list match the indexes of the `paragraphs` list and the value at that index contains the index of the table in the `tables` list that the paragraph belongs to.
        If the paragraph is not intersecting with any table, the index will be -1.

        For example, this assignments: [-1, 0, 0, 1, -1, -1, -1, 2, 2, 2, -1] means:
        The paragraph at index 0 does not belong to any table.
        The paragraphs at indexes 1 and 2 belong to table at index 0.
        The paragraph at index 3 belongs to table 1
        The rest of the paragraphs in the example belong to the table at index 2, or do not belong to any table
        """
        paragraph_to_table = [-1] * len(paragraphs)

        for paragraph_index, paragraph in enumerate(paragraphs):
            for table_index, table in enumerate(tables):
                if self._is_intersecting_regions(
                    paragraph.bounding_regions, table.bounding_regions
                ):
                    paragraph_to_table[paragraph_index] = table_index
                else:
                    continue

        return paragraph_to_table

    def _convert_to_paragraph(self, table):
        content = self._format_table(table)
        return DocumentParagraph(
            content=content, bounding_regions=table.bounding_regions, role="table"
        )

    def _format_table(self, table):
        """
        Formats Azure Document Intelligence's tables to the following format:
        <Table Captions>
        <Header1>: <ValueRow1>, <Header2>: <ValueRow1>, <Header3>: <ValueRow1>, ...
        <Header1>: <ValueRow2>, <Header2>: <ValueRow2>, <Header3>: <ValueRow2>, ...
        <Header1>: <ValueRow3>, <Header2>: <ValueRow3>, <Header3>: <ValueRow3>, ...
        ...
        """
        table_headers = []
        for cell in table["cells"]:
            if cell.get("kind") == "columnHeader":
                table_headers.append(cell["content"])

        content = table.get("caption", {}).get("content", "")

        previous_row_index = -1
        for cell in table.cells:
            if cell.get("kind") == "columnHeader":
                continue

            header = (
                f"{table_headers[cell['columnIndex']]}: "
                if cell["columnIndex"] < len(table_headers)
                else ""
            )

            is_new_row = previous_row_index != cell["rowIndex"]
            if is_new_row:
                content += "\n" if content else ""
                previous_row_index = cell["rowIndex"]

            content += f"{header}{cell.content}"
            content += ", " if cell["columnIndex"] < len(table_headers) - 1 else " "
        return content

    def _substitute_table_paragraphs(self, paragraphs, tables):
        """
        Returns a modified version of the `paragraphs` list, where paragraphs that are part of a table are combined and replaced with a formatted table.
        """
        result = []
        paragraphs_to_tables = self._assign_tables_to_paragraphs(paragraphs, tables)

        last_table_index = None
        for paragraph_index, table_index in enumerate(paragraphs_to_tables):
            is_table = table_index != -1
            if not is_table:
                result.append(paragraphs[paragraph_index])
                continue

            is_new_table = table_index != last_table_index
            if is_new_table:
                table = tables[table_index]
                formatted_table = self._convert_to_paragraph(table)
                result.append(formatted_table)
                last_table_index = table_index

        return result

    def _split_paragraphs_by_page(self, paragraphs):
        paragraphs_by_page = {}
        for paragraph in paragraphs:
            page_number = paragraph.bounding_regions[0]["pageNumber"]
            is_new_page = page_number not in paragraphs_by_page
            if is_new_page:
                paragraphs_by_page[page_number] = []
            paragraphs_by_page[page_number].append(paragraph)
        return paragraphs_by_page

    def _load_with_langchain(self, file_path, api_model):
        """
        Loads a file with LangChain's simpler implementation which returns the raw response from Document Intelligence.
        """

        documents = []
        try:
            loader = AzureAIDocumentIntelligenceLoader(
                file_path=file_path,
                api_key=self.key,
                api_endpoint=self.endpoint,
                api_model=api_model,
            )
            doc = loader.load()[0]
            doc.metadata = {
                "source": file_path,
                "page": 0,  # Azure Document Intelligence always returns a single page so we set it to 0
            }
            documents.append(doc)
        except Exception as e:
            logger.error(
                f"Failed to load {file_path} with Azure Document Intelligence using the 'prebuilt-read' model: {e}"
            )
            raise e

        logger.info(
            f'Successfully loaded {file_path} with Azure Document Intelligence using the "prebuilt-read" model.'
        )
        return documents
