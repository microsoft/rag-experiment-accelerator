from rag_experiment_accelerator.artifact.models.artifact import Artifact


def test_to_dict():
    artifact = Artifact()
    artifact_dict = artifact.to_dict()
    assert artifact_dict == {}
