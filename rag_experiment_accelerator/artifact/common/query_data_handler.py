from rag_experiment_accelerator.artifact.common.base import Base


class QueryOutputHandler(Base):
    """
    A class that handles the output of query data.

    Args:
        directory (str): The directory where the output files will be stored.
        **kwargs: Additional keyword arguments.

    Attributes:
        directory (str): The directory where the output files will be stored.

    Methods:
        get_output_filename(index_name: str) -> str:
            Returns the filename for the output file corresponding to the given index name.

        get_output_filepath(index_name: str) -> str:
            Returns the filepath for the output file corresponding to the given index name.
    """

    def __init__(self, directory: str, **kwargs) -> None:
        self.directory = directory
        super().__init__(
            directory=directory,
            **kwargs,
        )

    def get_output_filename(self, index_name: str) -> str:
        """
        Returns the filename for the output file corresponding to the given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The filename for the output file.
        """
        return f"eval_output_{index_name}.jsonl"

    def get_output_filepath(self, index_name: str) -> str:
        """
        Returns the filepath for the output file corresponding to the given index name.

        Args:
            index_name (str): The name of the index.

        Returns:
            str: The filepath for the output file.
        """
        return f"{self.directory}/{self.get_output_filename(index_name)}"
