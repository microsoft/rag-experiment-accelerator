import json
import os
import pathlib
import shutil
import uuid
import pytest

from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.artifact import Artifact
from rag_experiment_accelerator.writers.local.jsonl_writer import (
    JsonlWriter,
)


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_save_artifact(temp_dirname: str):
    writer = ArtifactWriter(temp_dirname, writer=JsonlWriter())
    a = Artifact()
    filename = "test.jsonl"
    filepath = f"{temp_dirname}/{filename}"

    # write the file
    writer.save_artifact(a, filename)

    # file exists with the correct data
    assert pathlib.Path(f"{filepath}").exists()
    with open(filepath, "r") as file:
        for line in file:
            data = json.loads(line)
            assert data == a.to_dict()


def test_archive_artifact(temp_dirname: str):
    writer = ArtifactWriter(temp_dirname, writer=JsonlWriter())
    # create the file to archive
    a = Artifact()
    os.makedirs(temp_dirname)
    path = f"{temp_dirname}/test.jsonl"
    with open(path, "w") as file:
        file.write(json.dumps(a.to_dict()) + "\n")

    # archive the artifact
    archive_filepath = writer.archive_artifact("test.jsonl")

    # archive dir exists
    assert pathlib.Path(f"{temp_dirname}/archive").exists()
    # archive file existså
    assert pathlib.Path(archive_filepath).exists()
    # original file does not exist
    assert not pathlib.Path(f"{temp_dirname}/test.jsonl").exists()
