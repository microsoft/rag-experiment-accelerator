import os
import pickle

from typing import Any, Dict, List, Set, Tuple
from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint


class LocalStorageCheckpoint(Checkpoint):
    def __init__(self, checkpoint_name: str, directory: str):
        self.checkpoint_location = f"{directory}/checkpoints/{checkpoint_name}"
        os.makedirs(self.checkpoint_location, exist_ok=True)
        self.internal_ids: Set[str] = self._get_existing_checkpoint_ids()

    def get_saved_ids(self, method) -> Set[str]:
        return set(
            [id.split("___")[1] for id in self.internal_ids if method.__name__ in id]
        )

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
        return f"{method.__name__}___{id}"

    def _get_existing_checkpoint_ids(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        ids = set()
        file_names = os.listdir(self.checkpoint_location)

        for file_name in file_names:
            file_name = file_name.replace(".pkl", "")
            ids.add(file_name)

        return ids
