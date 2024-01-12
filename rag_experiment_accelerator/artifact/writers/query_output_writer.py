from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.writers.local.jsonl_writer import (
    JsonlWriter,
)


class QueryOutputWriter(ArtifactWriter[QueryOutput], QueryOutputHandler):
    def __init__(self, output_dir: str) -> None:
        super().__init__(
            directory=output_dir,
            writer=JsonlWriter(),
        )

    def handle_archive(self, index_name: str):
        output_filename = self.get_output_filename(index_name)
        return super().archive_artifact(output_filename)

    def save(self, data: QueryOutput, index_name: str):
        output_filename = self.get_output_filename(index_name)
        super().save_artifact(data, output_filename)
