import argparse

from azure.ai.ml import MLClient
import mlflow

from rag_experiment_accelerator.run.evaluation import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import (
    mlflow_run_name,
    formatted_datetime_suffix,
)
from rag_experiment_accelerator.utils.auth import get_default_az_cred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="input: path to the input data",
        default=None,  # default is initialised in Config
    )
    args, _ = parser.parse_known_args()

    environment = Environment.from_env()
    config = Config(environment, args.config_path, args.data_dir)

    ml_client = MLClient(
        get_default_az_cred(),
        environment.aml_subscription_id,
        environment.aml_resource_group_name,
        environment.aml_workspace_name,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(config.NAME_PREFIX)
    mlflow_client = mlflow.MlflowClient(mlflow_tracking_uri)
    name_suffix = formatted_datetime_suffix()

    with mlflow.start_run(run_name=mlflow_run_name(config, name_suffix)):
        for index_config in config.index_configs():
            run(
                environment,
                config,
                index_config,
                mlflow_client,
                name_suffix=name_suffix,
            )
