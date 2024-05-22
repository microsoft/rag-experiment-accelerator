from promptflow import tool
from rag_experiment_accelerator.checkpoint import create_checkpoint
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_file_paths
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.config import Config


@tool
def my_python_tool(should_index: bool, config_path: str) -> bool:
    environment = Environment.from_env_or_keyvault()
    config = Config(environment, config_path)
    create_checkpoint(
        config.execution_environment, config.use_checkpoints, config.artifacts_dir
    )

    if should_index:
        file_paths = get_all_file_paths(config.data_dir)
        for index_config in config.index_configs():
            run(environment, config, index_config, file_paths)
    return True
