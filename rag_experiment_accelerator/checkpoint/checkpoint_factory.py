from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)

_checkpoint_instance = None


def create_checkpoint(
    type: str = "local",
    enable_checkpoints: bool = True,
    checkpoints_directory: str = "./artifacts",
):
    """
    Returns a Checkpoint object based on the provided configuration.
    """
    global _checkpoint_instance

    # Currently not supported in Azure ML: https://github.com/microsoft/rag-experiment-accelerator/issues/491
    if not enable_checkpoints or type != "local":
        _checkpoint_instance = NullCheckpoint()
    else:
        _checkpoint_instance = LocalStorageCheckpoint(directory=checkpoints_directory)

    return _checkpoint_instance


def get_checkpoint_instance():
    """
    Returns the instance of the Checkpoint object.
    """
    global _checkpoint_instance
    if _checkpoint_instance is None:
        raise Exception(
            "Checkpoint not initialized yet. Call CheckpointFactory.create_checkpoint() first."
        )
    return _checkpoint_instance


def reset_checkpoint_instance():
    """
    Resets the instance of the Checkpoint object.
    """
    global _checkpoint_instance
    _checkpoint_instance = None
