from promptflow import tool
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.paths import get_all_files
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config import Config


@tool
def my_python_tool(should_index: bool, config_path: str) -> bool:
    environment = Environment.from_env()
    config = Config(environment, config_path)

    if should_index:
        file_paths = get_all_files(config.data_dir)
        for index_config in config.index_configs():
            run(environment, config, index_config, file_paths)
    return True
