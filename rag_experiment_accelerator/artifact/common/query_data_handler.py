from rag_experiment_accelerator.artifact.common.base import Base


class QueryOutputHandler(Base):
    def __init__(self, directory: str, **kwargs) -> None:
        self.directory = directory
        super().__init__(
            directory=directory,
            **kwargs,
        )

    def get_output_filename(self, index_name: str) -> str:
        return f"eval_output_{index_name}.jsonl"

    def get_output_filepath(self, index_name: str) -> str:
        return f"{self.directory}/{self.get_output_filename(index_name)}"
