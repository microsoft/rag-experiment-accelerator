import argparse

import mlflow

from azureml.pipeline import initialise_mlflow_client
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.run.evaluation import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.paths import (
    mlflow_run_name,
    formatted_datetime_suffix,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="input: path to the input data",
        default=None,  # default is initialized in Config
    )
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    config = Config.from_path(environment, args.config_path, args.data_dir)
    name_suffix = formatted_datetime_suffix()
    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    for index_config in config.index_config.flatten():
        with mlflow.start_run(run_name=mlflow_run_name(config.job_name, name_suffix)):
            run(
                environment,
                config,
                index_config,
                mlflow_client,
                name_suffix=name_suffix,
            )
