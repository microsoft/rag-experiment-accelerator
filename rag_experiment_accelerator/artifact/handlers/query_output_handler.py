from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.io.local.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.io.local.writers.jsonl_writer import JsonlWriter


class QueryOutputHandler(ArtifactHandler):
    def __init__(self, data_location: str) -> None:
        super().__init__(
            data_location=data_location, writer=JsonlWriter(), loader=JsonlLoader()
        )

    def _get_output_name(self, index_name: str) -> str:
        return f"eval_output_{index_name}.jsonl"

    def get_output_path(self, index_name: str) -> str:
        return f"{self.data_location}/{self._get_output_name(index_name)}"

    def load(self, index_name: str) -> list[QueryOutput]:
        query_outputs = []
        filename = self._get_output_name(index_name)
        data_load = super().load(filename)
        for d in data_load:
            d = QueryOutput(**d)
            query_outputs.append(d)
        return query_outputs

    def handle_archive_by_index(self, index_name: str) -> str | None:
        output_filename = self._get_output_name(index_name)
        return super().handle_archive(output_filename)

    def save(self, data: QueryOutput, index_name: str):
        output_filename = self._get_output_name(index_name)
        self.save_dict(data.__dict__, output_filename)
