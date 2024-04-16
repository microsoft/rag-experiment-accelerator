from abc import ABC, abstractmethod
from typing import Any, Set
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class Checkpoint(ABC):
    """
    A Checkpoint object is used to recover and continue the execution of previous executions.
    A Checkpoint is identified by the checkpoint's name and the config (name) that it was created with.

    A checkpoint object is used to wrap methods, so when the method is called with an ID that was called before,
    instead of executing the method, the checkpoint will return the result of the previous execution.
    """

    @abstractmethod
    def __init__(self, checkpoint_name: str, config_name: str, config: Config):
        pass

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
                f"Checkpoint data found for '{method.__name__}' - skipped execution and loaded from checkpoint. ID:'{id}'"
            )
            return self._load(id, method)
        else:
            method_result = method(*args, **kwargs)
            self._save(method_result, id, method)
            return method_result

    @abstractmethod
    def get_ids(self, method) -> Set[str]:
        """
        Returns a set of the IDs for which there are checkpoints available for the given method.
        """
        pass

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
