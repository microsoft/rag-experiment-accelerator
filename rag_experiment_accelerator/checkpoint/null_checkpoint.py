from typing import Any
from rag_experiment_accelerator.checkpoint import Checkpoint


class NullCheckpoint(Checkpoint):
    """
    A dummy checkpoint implementation that does not do anything, used in cases where the checkpoints should be ignored.
    """

    def __init__(self):
        pass

    def _has_data(self, id: str, method) -> bool:
        return False

    def _load(self, id: str, method) -> Any:
        pass

    def _save(self, data: Any, id: str, method):
        pass
