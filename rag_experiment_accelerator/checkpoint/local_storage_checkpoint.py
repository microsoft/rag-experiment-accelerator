import os
import pickle
import hashlib

from typing import Any, List, Set
from rag_experiment_accelerator.checkpoint import Checkpoint


class LocalStorageCheckpoint(Checkpoint):
    """
    A checkpoint implementation that stores the data in the local file system.
    """

    def __init__(self, checkpoint_name: str, directory: str = "."):
        self.checkpoint_location = f"{directory}/checkpoints/{checkpoint_name}"
        os.makedirs(self.checkpoint_location, exist_ok=True)
        self.internal_ids: Set[str] = self._get_existing_checkpoint_ids()

    def _has_data(self, id: str, method) -> bool:
        checkpoint_id = self._build_internal_id(id, method)
        return checkpoint_id in self.internal_ids

    def _load(self, id: str, method) -> List:
        file_path = self._get_checkpoint_file_path(id, method)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            return data

    def _save(self, data: Any, id: str, method):
        file_path = self._get_checkpoint_file_path(id, method)
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        internal_id = self._build_internal_id(id, method)
        self.internal_ids.add(internal_id)

    def _get_checkpoint_file_path(self, id: str, method):
        checkpoint_id = self._build_internal_id(id, method)
        return f"{self.checkpoint_location}/{checkpoint_id}.pkl"

    def _build_internal_id(self, id: str, method):
        hashed_id = hashlib.sha256(id.encode()).hexdigest()
        return f"{method.__name__}___{hashed_id}"

    def _get_existing_checkpoint_ids(self) -> Set[str]:
        ids = set()
        file_names = os.listdir(self.checkpoint_location)

        for file_name in file_names:
            file_name = file_name.replace(".pkl", "")
            ids.add(file_name)

        return ids
