from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from tqdm import tqdm
import re
import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document
from pathlib import Path

from langchain_community.document_loaders.base import BaseLoader
from typing import List, Iterator
from rag_experiment_accelerator.utils.logging import get_logger
from azure.ai.documentintelligence.models import DocumentParagraph

logger = get_logger(__name__)


class DocumentIntelligenceLoader(BaseLoader):
    """
    Analyzes and loads documents and directories using Azure Document Intelligence.
    """

    def __init__(
        self,
        path: str,
        endpoint: str,
        key: str,
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
        self.patterns_to_remove = patterns_to_remove
        self.glob_patterns = glob_patterns
        self.split_documents_by_page = split_documents_by_page
        self.excluded_paragraph_roles = excluded_paragraph_roles

    def load(self) -> List[Document]:
        documents = []
        file_paths = self._get_file_paths()

        with ExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor())
            progress_bar = stack.enter_context(
                tqdm(total=len(file_paths), desc="Analyzing documents")
            )

            futures = {
                executor.submit(self._analyze_document, file_path)
                for file_path in file_paths
            }

            for future in as_completed(futures):
                try:
                    documents += future.result()
                except Exception as exc:
                    logger.error(f"Processing document generated an exception: {exc}")
                progress_bar.update(1)

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
        documents = []
        try:
            with open(file_path, "rb") as file:
                content = file.read()
                poller = self.client.begin_analyze_document(
                    "prebuilt-layout",
                    content,
                    content_type="application/octet-stream",
                    output_content_format="markdown",
                )

            result = poller.result()
            paragraphs = self._substitute_table_paragraphs(
                result.paragraphs, result.tables
            )

            relevant_paragraphs = [
                paragraph
                for paragraph in paragraphs
                if paragraph.role not in self.excluded_paragraph_roles
            ]
            paragraphs_by_role = self._get_paragraphs_by_role(result)

            if self.split_documents_by_page:
                paragraphs_by_page = self._split_paragraphs_by_page(relevant_paragraphs)
                for page_number, page_paragraphs in paragraphs_by_page.items():
                    documents.append(
                        self._convert_to_langchain_document(
                            page_paragraphs, file_path, paragraphs_by_role, page_number
                        )
                    )
            else:
                documents.append(
                    self._convert_to_langchain_document(
                        relevant_paragraphs, file_path, paragraphs_by_role, 1
                    )
                )

            return documents
        except Exception as exc:
            logger.warning(
                f"Failed to load {file_path} with Azure Document Intelligence using the 'prebuilt-layout' model: {exc}. Attempting to load using the simpler 'prebuilt-read' model..."
            )
            return self._load_with_ocr(file_path)

    def _clean_content(self, content: str):
        # Remove AI doc intelligence traces.
        pattern = re.compile(r":selected:|:unselected:")
        content = pattern.sub("", content)
        # Remove specific regex patterns.
        for regex_pattern in self.patterns_to_remove:
            content = regex_pattern.sub("", content)

        return content

    def _get_paragraphs_by_role(self, result):
        dict = {}
        for paragraph in result.paragraphs:
            if not paragraph.role or paragraph.role in self.excluded_paragraph_roles:
                continue
            paragraph_item = {
                "content": paragraph.content,
                "page": paragraph.bounding_regions[0].get("pageNumber"),
            }
            dict[paragraph.role] = dict.get(paragraph.role, []) + [paragraph_item]

        tables = []
        for table in result.tables:
            table_item = {
                "cells": table.cells,
                "page": table.bounding_regions[0].get("pageNumber"),
            }
            tables.append(table_item)
        dict["tables"] = tables

        return dict

    def _convert_to_langchain_document(
        self, paragraphs, file_path, paragraphs_by_role, page_number
    ):
        content = "\n\n".join([paragraph.content for paragraph in paragraphs])
        clean_content = self._clean_content(content)
        return Document(
            page_content=clean_content,
            metadata={
                "source": file_path,
                "paragraphs_by_role": paragraphs_by_role,
                "page": page_number - 1,
            },
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
        rows_to_span = 0
        spanning_content = ""
        for cell in table.cells:
            if cell.get("kind") == "columnHeader":
                continue

            header = (
                f"{table_headers[cell['columnIndex']]}: "
                if cell["columnIndex"] < len(table_headers)
                else ""
            )

            # If the cell spans multiple rows, we need to combine the content of the spanning cells
            if rows_to_span > 0:
                spanning_content += cell.content
                rows_to_span -= 1
                if rows_to_span == 0:
                    content += f"{header}{spanning_content}"
                    spanning_content = ""
                else:
                    spanning_content += ", "
                continue
            else:
                rows_to_span = cell.get("rowSpan", 0)

            is_new_row = previous_row_index != cell["rowIndex"]
            if is_new_row:
                content += "\n" if content else ""
                previous_row_index = cell["rowIndex"]

            content += f"{header}{cell.content}"
            content += ", " if cell["columnIndex"] < len(table_headers) - 1 else ""
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

    def _load_with_ocr(self, file_path):
        """
        Loads a file with a simpler 'prebuilt-read' model which uses a simple OCR approach to load the file.
        Some files may not be supported by the 'prebuilt-layout' model, but can be loaded with the 'prebuilt-read' model.
        """

        document = []
        try:
            loader = AzureAIDocumentIntelligenceLoader(
                file_path=file_path,
                api_key=self.key,
                api_endpoint=self.endpoint,
                api_model="prebuilt-read",
            )
            document += loader.load()
        except Exception as e:
            logger.error(
                f"Failed to load {file_path} with Azure Document Intelligence using the 'prebuilt-read' model: {e}"
            )

        logger.info(
            f'Successfully loaded {file_path} with Azure Document Intelligence using the "prebuilt-read" model.'
        )
        return document
