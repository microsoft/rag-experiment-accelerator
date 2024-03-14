from promptflow import tool
import mlflow

from rag_experiment_accelerator.run.evaluation import run, initialise_mlflow_client
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.config.paths import (
    mlflow_run_name,
    formatted_datetime_suffix,
)


@tool
def my_python_tool(config_path: str) -> bool:
    environment = Environment.from_env()
    config = Config(environment, config_path)
    mlflow_client = initialise_mlflow_client(environment, config)
    name_suffix = formatted_datetime_suffix()

    with mlflow.start_run(run_name=mlflow_run_name(config, name_suffix)):
        for index_config in config.index_configs():
            run(
                environment,
                config,
                index_config,
                mlflow_client,
                formatted_datetime_suffix,
            )
    return True
