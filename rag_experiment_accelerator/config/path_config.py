from dataclasses import dataclass
import os
from rag_experiment_accelerator.config import paths
from rag_experiment_accelerator.config.base_config import BaseConfig


class Paths:
    ARTIFACTS_DIR = "artifacts"
    DATA_DIR = "data"
    EVAL_DATA_FILE = "eval_data.jsonl"
    GENERATED_INDEX_NAMES_FILE = "generated_index_names.jsonl"
    QUERY_DATA_DIR = "query_data"
    EVAL_DATA_DIR = "eval_score"
    SAMPLING_OUTPUT_DIR = "sampling"


@dataclass
class PathConfig(BaseConfig):
    artifacts_dir: str = ""
    data_dir: str = ""
    eval_data_file: str = ""
    eval_data_dir: str = ""
    generated_index_names_file: str = ""
    query_data_dir: str = ""
    sampling_output_dir: str = ""

    def initialize_paths(self, config_file_path: str, data_dir: str) -> None:
        self._config_dir = os.path.dirname(config_file_path)

        if not self.artifacts_dir:
            self.artifacts_dir = os.path.join(self._config_dir, Paths.ARTIFACTS_DIR)
        paths.try_create_directory(self.artifacts_dir)

        if data_dir:
            self.data_dir = data_dir
        elif not self.data_dir:
            self.data_dir = os.path.join(self._config_dir, Paths.DATA_DIR)

        if not self.eval_data_file:
            self.eval_data_file = os.path.join(self.artifacts_dir, Paths.EVAL_DATA_FILE)

        if not self.generated_index_names_file:
            self.generated_index_names_file = os.path.join(
                self.artifacts_dir, Paths.GENERATED_INDEX_NAMES_FILE
            )

        if not self.query_data_dir:
            self.query_data_dir = os.path.join(self.artifacts_dir, Paths.QUERY_DATA_DIR)
        paths.try_create_directory(self.query_data_dir)

        if not self.eval_data_dir:
            self.eval_data_dir = os.path.join(self.artifacts_dir, Paths.EVAL_DATA_DIR)
        paths.try_create_directory(self.eval_data_dir)

        if not self.sampling_output_dir:
            self.sampling_output_dir = os.path.join(
                self.artifacts_dir, Paths.SAMPLING_OUTPUT_DIR
            )
        paths.try_create_directory(self.sampling_output_dir)

    def sampled_cluster_predictions_path(self, optimum_k: int) -> str:
        return os.path.join(
            self.sampling_output_dir,
            f"sampled_cluster_predictions_cluster_number_{optimum_k}.csv",
        )
