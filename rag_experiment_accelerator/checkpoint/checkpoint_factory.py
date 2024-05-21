from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)

global _instance
_instance = None


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

        global _instance
        _instance = LocalStorageCheckpoint(directory=checkpoints_directory)

        return _instance

    @staticmethod
    def get_instance():
        """
        Returns the instance of the Checkpoint object.
        """
        global _instance
        if _instance is None:
            raise Exception(
                "Checkpoint not initialized yet. Call CheckpointFactory.create_checkpoint() first."
            )
        return _instance

    @staticmethod
    def reset():
        """
        Resets the instance of the Checkpoint object.
        """
        global _instance
        _instance = None
