from promptflow import tool
from rag_experiment_accelerator.run.querying import run
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.config import Config


@tool
def my_python_tool(config_path: str) -> bool:
    environment = Environment.from_env_or_keyvault()
    config = Config.from_path(environment, config_path)

    for index_config in config.index_config.flatten():
        run(environment, config, index_config)
    return True
