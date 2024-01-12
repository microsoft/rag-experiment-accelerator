from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.writers.local.jsonl_writer import (
    JsonlWriter,
)


class QueryOutputWriter(ArtifactWriter[QueryOutput], QueryOutputHandler):
    """
    A class that writes query output data to a specified directory.

    Args:
        output_dir (str): The directory where the output files will be saved.

    Attributes:
        directory (str): The directory where the output files will be saved.
        writer (ArtifactWriter): The writer used to write the output files.

    """

    def __init__(self, output_dir: str) -> None:
        super().__init__(
            directory=output_dir,
            writer=JsonlWriter(),
        )

    def handle_archive(self, index_name: str) -> str | None:
        """
        Archives the artifact for the given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str | None: The filename of the archived artifact, or None if archiving failed.

        """
        output_filename = self.get_output_filename(index_name)
        return super().archive_artifact(output_filename)

    def save(self, data: QueryOutput, index_name: str):
        """
        Saves the query output data for the given index name.

        Args:
            data (QueryOutput): The query output data to be saved.
            index_name (str): The name of the index.

        """
        output_filename = self.get_output_filename(index_name)
        super().save_artifact(data, output_filename)
