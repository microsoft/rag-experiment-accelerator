import os
import pickle

from typing import Any, List
from rag_experiment_accelerator.checkpoint.checkpoint import Checkpoint
from rag_experiment_accelerator.config.config import Config


class LocalStorageCheckpoint(Checkpoint):
    def __init__(self, checkpoint_name: str, config_name: str, config: Config):
        self.config = config
        self.checkpoint_file_name = f"{checkpoint_name}_{config_name}.pkl"
        self.checkpoint_location = (
            f"{config.artifacts_dir}/checkpoints/{self.checkpoint_file_name}"
        )
        os.makedirs(f"{self.config.artifacts_dir}/checkpoints", exist_ok=True)

    def exists(self) -> bool:
        return os.path.exists(self.checkpoint_location)

    def load(self) -> List:
        with open(self.checkpoint_location, "rb") as file:
            data = pickle.load(file)
            return [data]

    def save(self, data: Any):
        with open(self.checkpoint_location, "wb") as file:
            pickle.dump(data, file)
