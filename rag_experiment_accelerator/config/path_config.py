from rag_experiment_accelerator.config.base_config import BaseConfig


class PathConfig(BaseConfig):
    artifacts_dir: str | None = None
    data_dir: str | None = None
