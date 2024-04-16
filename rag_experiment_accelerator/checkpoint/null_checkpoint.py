from typing import Any, Set
from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint


class NullCheckpoint(Checkpoint):
    def __init__(self):
        pass

    def get_ids(self, method) -> Set[str]:
        return set()

    def _has_data(self, id: str, method) -> bool:
        return False

    def _load(self, id: str, method) -> Any:
        pass

    def _save(self, data: Any, id: str, method):
        pass
