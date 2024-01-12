from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.loaders.local.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.writers.local.jsonl_writer import JsonlWriter


class QueryOutputHandler(ArtifactHandler):
    def __init__(self, output_dir: str) -> None:
        super().__init__(
            directory=output_dir, writer=JsonlWriter(), loader=JsonlLoader()
        )

    def get_output_filename(self, index_name: str) -> str:
        return f"eval_output_{index_name}.jsonl"

    def get_output_filepath(self, index_name: str) -> str:
        return f"{self.directory}/{self.get_output_filename(index_name)}"

    def load(self, index_name: str) -> list[QueryOutput]:
        query_outputs = []
        path = self.get_output_filepath(index_name)
        data_load = self._loader.load(path)
        for d in data_load:
            d = QueryOutput(**d)
            query_outputs.append(d)
        return query_outputs

    def archive(self, index_name: str) -> str | None:
        output_filename = self.get_output_filename(index_name)
        return super().archive(output_filename)

    def save(self, data: QueryOutput, index_name: str):
        output_filename = self.get_output_filename(index_name)
        path = f"{self.directory}/{output_filename}"
        self._writer.write(path, data.__dict__)
