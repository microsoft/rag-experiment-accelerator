from promptflow import tool
from rag_experiment_accelerator.checkpoint import init_checkpoint
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_file_paths
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.config import Config


@tool
def my_python_tool(should_index: bool, config_path: str) -> bool:
    environment = Environment.from_env_or_keyvault()
    config = Config(environment, config_path)
    init_checkpoint(config)

    if should_index:
        file_paths = get_all_file_paths(config.path.data_dir)
        for index_config in config.index_config.flatten():
            run(environment, config, index_config, file_paths)
    return True
