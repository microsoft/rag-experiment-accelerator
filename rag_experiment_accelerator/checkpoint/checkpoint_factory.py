from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)


class CheckpointFactory:
    """
    A factory class that creates a Checkpoint object based on the provided configuration.
    """

    @staticmethod
    def create_checkpoint(
        type: str = "local",
        enable_checkpoints: bool = True,
        checkpoints_directory: str = "./artifacts",
    ):
        """
        Returns a Checkpoint object based on the provided configuration.
        """
        if not enable_checkpoints:
            return NullCheckpoint()

        if type == "azure-ml":
            # Currently not supported in Azure ML: https://github.com/microsoft/rag-experiment-accelerator/issues/491
            return NullCheckpoint()

        return LocalStorageCheckpoint(directory=checkpoints_directory)
