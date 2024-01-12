import json

from rag_experiment_accelerator.writers.local.local_writer import LocalWriter


class JsonlWriter(LocalWriter):
    def _write(self, path: str, data, **kwargs):
        with open(path, "a") as file:
            file.write(json.dumps(data, **kwargs) + "\n")
