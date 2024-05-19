from rag_experiment_accelerator.config.config import Config, ExecutionEnvironment

global _checkpoint_instance
_checkpoint_instance = None


def get_checkpoint():
    """
    Returns the current checkpoint instance.
    """
    global _checkpoint_instance
    if not _checkpoint_instance:
        raise Exception("Checkpoint not initialized yet. Call init_checkpoint() first.")
    return _checkpoint_instance


def init_checkpoint(config: Config):
    """
    Initializes the checkpoint instance based on the provided configuration.
    """
    global _checkpoint_instance
    _checkpoint_instance = _get_checkpoint_base_on_config(config)


def _get_checkpoint_base_on_config(config: Config):
    # import inside the method to avoid circular dependencies
    from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
    from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
        LocalStorageCheckpoint,
    )

    if not config.use_checkpoints:
        return NullCheckpoint()

    if config.execution_environment == ExecutionEnvironment.AZURE_ML:
        # Currently not supported in Azure ML: https://github.com/microsoft/rag-experiment-accelerator/issues/491
        return NullCheckpoint()

    return LocalStorageCheckpoint(directory=config.artifacts_dir)
