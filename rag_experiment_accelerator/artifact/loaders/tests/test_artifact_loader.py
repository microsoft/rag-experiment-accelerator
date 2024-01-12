import os
import shutil
import tempfile
import pytest

from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.artifact import Artifact
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.loaders.local.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.writers.local.jsonl_writer import (
    JsonlWriter,
)


@pytest.fixture()
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_loads(temp_dir: str):
    # # write artifacts to a file
    writer = ArtifactWriter(temp_dir, writer=JsonlWriter())
    artifacts = [Artifact(), Artifact()]
    filename = "test.jsonl"
    for artifact in artifacts:
        writer.save_artifact(artifact, filename)

    # load the file
    loader = ArtifactLoader(
        class_to_load=Artifact, directory=temp_dir, loader=JsonlLoader()
    )
    loaded_data = loader.load_artifacts(filename)

    assert [a.to_dict() for a in loaded_data] == [a.to_dict() for a in artifacts]
    assert [isinstance(a, Artifact) for a in loaded_data]
