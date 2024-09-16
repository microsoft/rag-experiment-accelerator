from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.artifact.handlers.typing import T, U
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.io.local.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.io.local.writers.jsonl_writer import JsonlWriter


class QueryOutputHandler(ArtifactHandler):
    """
    A class that handles query outputs for a given index name.
    """

    def __init__(
        self, data_location: str, writer: T = JsonlWriter(), loader: U = JsonlLoader()
    ) -> None:
        """
        Initializes a QueryOutputHandler instance.

        Args:
            data_location (str): The location where the data is stored.
            writer (T, optional): The writer to use for saving data. Defaults to JsonlWriter().
            loader (U, optional): The loader to use for loading data. Defaults to JsonlLoader().
        """
        super().__init__(data_location=data_location, writer=writer, loader=loader)

    def _get_output_name(
        self, index_name: str, experiment_name: str, job_name: str
    ) -> str:
        """
        Returns the output name for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The output name.
        """
        return f"eval_output_{index_name}_{experiment_name}_{job_name}.jsonl"

    def get_output_path(
        self, index_name: str, experiment_name: str, job_name: str
    ) -> str:
        """
        Returns the output path for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The output path.
        """
        return f"{self.data_location}/{self._get_output_name(index_name, experiment_name, job_name)}"

    def load(
        self, index_name: str, experiment_name: str, job_name: str
    ) -> list[QueryOutput]:
        """
        Loads the query outputs for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            list[QueryOutput]: The loaded query outputs.
        """
        output_name = self._get_output_name(index_name, experiment_name, job_name)

        query_outputs = []
        data_load = super().load(output_name)
        for d in data_load:
            if not isinstance(d, dict):
                raise TypeError(
                    f"Query output data loaded is not of type dict. Name: {output_name}"
                )
            query_outputs.append(QueryOutput(**d))

        return query_outputs

    def handle_archive_by_index(
        self, index_name: str, experiment_name: str, job_name: str
    ) -> str | None:
        """
        Handles archiving of query output for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str | None: The output filename if successful, None otherwise.
        """
        output_filename = self._get_output_name(index_name, experiment_name, job_name)
        return self.handle_archive(output_filename)

    def save(
        self, data: QueryOutput, index_name: str, experiment_name: str, job_name: str
    ):
        """
        Saves the query output for a given index name.

        Args:
            data (QueryOutput): The query output to be saved.
            index_name (str): The name of the index.
        """
        output_filename = self._get_output_name(index_name, experiment_name, job_name)
        self.save_dict(data.__dict__, output_filename)
