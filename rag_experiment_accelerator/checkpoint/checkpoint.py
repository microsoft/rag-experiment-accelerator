from abc import ABC, abstractmethod
from typing import Any
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class Checkpoint(ABC):
    """
    A Checkpoint is used to cache the results of method calls, enabling the reuse of these results if the same method is called again with the same ID.
    When a method wrapped by a Checkpoint object is called with an ID that was used before,
    the Checkpoint returns the result of the previous execution instead of executing the method again.
    """

    @abstractmethod
    def __init__(self, checkpoint_name: str, directory: str):
        """
        Initializes the checkpoint object.

        Parameters:
        - checkpoint_name (str): The name of the checkpoint, the checkpoint is uniquely identified by its name.
        - directory (str): The directory where the '/checkpoints' directory will be stored.
        """
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
