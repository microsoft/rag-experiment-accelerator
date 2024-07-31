from dataclasses import dataclass
import os
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class PathConfig(BaseConfig):
    artifacts_dir: str = ""
    data_dir: str = ""
    eval_data_file: str = ""
    eval_data_dir: str = ""
    generated_index_names_file: str = ""
    query_data_dir: str = ""
    eval_data_dir: str = ""
    sampling_output_dir: str = ""

    def initialize_paths(self, config_file_path: str, data_dir: str) -> None:
        self._config_dir = os.path.dirname(config_file_path)

        if not self.artifacts_dir:
            self.artifacts_dir = os.path.join(self._config_dir, "artifacts")
        self._try_create_directory(self.artifacts_dir)

        if data_dir:
            self.data_dir = data_dir
        elif not self.data_dir:
            self.data_dir = os.path.join(self._config_dir, "data")

        if not self.eval_data_file:
            self.eval_data_file = os.path.join(self.artifacts_dir, "eval_data.jsonl")

        if not self.generated_index_names_file:
            self.generated_index_names_file = os.path.join(
                self.artifacts_dir, "generated_index_names.jsonl"
            )
        if not self.query_data_dir:
            self.query_data_dir = os.path.join(self.artifacts_dir, "query_data")
        self._try_create_directory(self.query_data_dir)

        if not self.eval_data_dir:
            self.eval_data_dir = os.path.join(self.artifacts_dir, "eval_score")
        self._try_create_directory(self.eval_data_dir)

        if not self.sampling_output_dir:
            self.sampling_output_dir = os.path.join(self.artifacts_dir, "sampling")
        self._try_create_directory(self.sampling_output_dir)
