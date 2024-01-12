from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.loaders.local.jsonl_loader import JsonlLoader


class QueryOutputLoader(ArtifactLoader[QueryOutput], QueryOutputHandler):
    """
    A class for loading query output artifacts.

    This class extends the `ArtifactLoader` class and implements the `QueryOutputHandler` interface.
    It provides methods to load query output artifacts from a specified directory.

    Args:
        output_dir (str): The directory where the query output artifacts are stored.

    Attributes:
        output_dir (str): The directory where the query output artifacts are stored.

    """

    def __init__(self, output_dir: str) -> None:
        super().__init__(
            class_to_load=QueryOutput,
            directory=output_dir,
            loader=JsonlLoader(),
        )

    def load_all(self, index_name: str) -> list[QueryOutput]:
        """
        Load all query output artifacts for a given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            list[QueryOutput]: A list of loaded query output artifacts.

        """
        path = self.get_output_filename(index_name)
        return super().load_artifacts(path)
