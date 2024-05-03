from abc import ABC, abstractmethod
from typing import Any
from rag_experiment_accelerator.config.config import Config, ExecutionEnvironment
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

global _checkpoint_instance


def get_checkpoint():
    """
    Returns the current checkpoint instance.
    """
    global _checkpoint_instance
    if not _checkpoint_instance:
        raise RuntimeError(
            "Checkpoint not initialized yet. Call init_checkpoint() first."
        )
    return _checkpoint_instance


def init_checkpoint(checkpoint_name, config: Config):
    """
    Initializes the checkpoint instance based on the provided configuration.
    """
    global _checkpoint_instance
    _checkpoint_instance = _get_checkpoint_base_on_config(checkpoint_name, config)


def _get_checkpoint_base_on_config(checkpoint_name, config: Config):
    # import inside the method to avoid circular dependencies
    from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
    from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
        LocalStorageCheckpoint,
    )

    if not config.USE_CHECKPOINTS:
        return NullCheckpoint()

    if config.EXECUTION_ENVIRONMENT == ExecutionEnvironment.AZURE_ML:
        # currently not supposed for Azure ML: https://github.com/microsoft/rag-experiment-accelerator/issues/491
        return NullCheckpoint()

    return LocalStorageCheckpoint(
        checkpoint_name=checkpoint_name,
        directory=config.artifacts_dir,
    )


class Checkpoint(ABC):
    """
    A Checkpoint is used to cache the results of method calls, enabling the reuse of these results if the same method is called again with the same ID.
    When a method wrapped by a Checkpoint object is called with an ID that was used before,
    the Checkpoint returns the result of the previous execution instead of executing the method again.

    Initialize a Checkpoint using the `init_checkpoint` method, and use the `get_checkpoint` method to get the Checkpoint object.
    """

    def load_or_run(self, method, id: str, *args, **kwargs) -> Any:
        """
        Checks if the provided method has previously been executed with the given ID,
        If it has, it returns the cached result,
        otherwise, it executes the method with the given arguments and caches the result for future calls.

        Parameters:
        - id (str): A unique identifier for the data.
        - method: The method to be executed.
        - *args: Variable length argument list for the method.
        - **kwargs: Arbitrary keyword arguments for the method.

        Returns:
        - Any: The result of the method execution.
        """
        if self._has_data(id, method):
            logger.info(
                f"Checkpoint data found for '{method.__name__}' - skipping execution and loading from checkpoint."
            )
            return self._load(id, method)
        else:
            method_result = method(*args, **kwargs)
            self._save(method_result, id, method)
            return method_result

    @abstractmethod
    def _has_data(self, id: str, method) -> bool:
        """
        Returns whether the checkpoint has any data for the given method with the given id.

        Args:
        - id (str): A unique identifier for the data.
        - method: The method that is wrapped by the checkpoint.
        """
        pass

    @abstractmethod
    def _load(self, id: str, method) -> Any:
        """
        Loads data for the given id and method.

        Args:
            id (str): A unique identifier for the data.
            method: The method used to load the data.

        Returns:
            Any: The loaded data.
        """
        pass

    @abstractmethod
    def _save(self, data: Any, id: str, method):
        """
        Saves data to the checkpoint.

        Args:
            data (Any): A unique identifier for the data.
            id (str): The identifier for the data.
            method: The method used.
        """
        pass
