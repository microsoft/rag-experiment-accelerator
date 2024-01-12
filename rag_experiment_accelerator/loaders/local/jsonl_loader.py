import json
import os
import pathlib
from rag_experiment_accelerator.loaders.local.local_loader import LocalLoader


class JsonlLoader(LocalLoader):
    def load(self, path: str, **kwargs) -> list:
        data_load = []
        if os.path.exists(path):
            with open(path, "r") as file:
                for line in file:
                    data = json.loads(line, **kwargs)
                    data_load.append(data)
        return data_load

    def can_handle(self, path: str):
        ext = pathlib.Path(path).suffix
        return ext == ".jsonl"
