from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.artifact.handlers.typing import T, U
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.io.local.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.io.local.writers.jsonl_writer import JsonlWriter


class QueryOutputHandler(ArtifactHandler):
    """
    A class that handles query output artifacts.

    Args:
        data_location (str): The location where the data is stored.

    Attributes:
        data_location (str): The location where the data is stored.
        writer (JsonlWriter): The writer used for writing data.
        loader (JsonlLoader): The loader used for loading data.
    """

    def __init__(
        self, data_location: str, writer: T = JsonlWriter(), loader: U = JsonlLoader()
    ) -> None:
        """
        Initializes a QueryOutputHandler instance.

        Args:
            data_location (str): The location where the data is stored.
        """
        super().__init__(data_location=data_location, writer=writer, loader=loader)

    def _get_output_name(self, index_name: str) -> str:
        """
        Returns the output name for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The output name.
        """
        return f"eval_output_{index_name}.jsonl"

    def get_output_path(self, index_name: str) -> str:
        """
        Returns the output path for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The output path.
        """
        return f"{self.data_location}/{self._get_output_name(index_name)}"

    def load(self, index_name: str) -> list[QueryOutput]:
        """
        Loads the query outputs for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            list[QueryOutput]: The loaded query outputs.
        """
        filename = self._get_output_name(index_name)

        query_outputs = []
        data_load = super().load(filename)
        for d in data_load:
            d = QueryOutput(**d)
            query_outputs.append(d)

        return query_outputs

    def handle_archive_by_index(self, index_name: str) -> str | None:
        """
        Handles archiving of query output for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str | None: The output filename if successful, None otherwise.
        """
        output_filename = self._get_output_name(index_name)
        return self.handle_archive(output_filename)

    def save(self, data: QueryOutput, index_name: str):
        """
        Saves the query output for a given index name.

        Args:
            data (QueryOutput): The query output to be saved.
            index_name (str): The name of the index.
        """
        output_filename = self._get_output_name(index_name)
        self.save_dict(data.__dict__, output_filename)
