from abc import ABC, abstractmethod
from typing import List, Any
from rag_experiment_accelerator.config.config import Config


class Checkpoint(ABC):
    """
    A Checkpoint object is used to recover and continue the execution of previous executions.
    A Checkpoint is identified by the checkpoint's name and the config (name) that it was created with.
    """

    @abstractmethod
    def __init__(self, checkpoint_name: str, config_name: str, config: Config):
        pass

    @abstractmethod
    def exists(self) -> bool:
        """
        Returns whether the checkpoint has any data.
        """
        pass

    @abstractmethod
    def load(self) -> List:
        """
        Loads the checkpoint data into memory.

        Returns:
        List: The data stored in the checkpoint. Each element of the list represents a checkpoint from a different node.
        This is particularly relevant for distributed executions, such as those in AzureML. For local executions, the list will have only one element.
        """
        pass

    @abstractmethod
    def save(self, data: Any):
        """
        Saves data to the checkpoint.
        """
        pass
