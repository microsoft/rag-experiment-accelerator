from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.loaders.local.jsonl_loader import JsonlLoader


class QueryOutputLoader(ArtifactLoader[QueryOutput], QueryOutputHandler):
    def __init__(self, output_dir: str) -> None:
        super().__init__(
            class_to_load=QueryOutput,
            directory=output_dir,
            loader=JsonlLoader(),
        )

    def load_all(self, index_name: str) -> list[QueryOutput]:
        path = self.get_output_filename(index_name)
        return super().load_artifacts(path)
