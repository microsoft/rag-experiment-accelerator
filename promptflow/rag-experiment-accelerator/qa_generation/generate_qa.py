from promptflow import tool
from rag_experiment_accelerator.run.qa_generation import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import get_all_file_paths


@tool
def my_python_tool(config_path: str, should_generate_qa: bool) -> bool:
    environment = Environment.from_env_or_keyvault()
    config = Config(environment, config_path)

    if should_generate_qa:
        run(environment, config, get_all_file_paths(config.path.data_dir))
    return True
