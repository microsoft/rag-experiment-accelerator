import json

from rag_experiment_accelerator.io.local.loaders.local_loader import LocalLoader


class JsonlLoader(LocalLoader):
    def load(self, path: str, **kwargs) -> list:
        data_load = []
        if self.exists(path):
            with open(path, "r") as file:
                for line in file:
                    data = json.loads(line, **kwargs)
                    data_load.append(data)
        else:
            raise FileNotFoundError(f"File not found at path: {path}")
        return data_load

    def can_handle(self, path: str):
        ext = self._get_file_ext(path)
        return ext == ".jsonl"
