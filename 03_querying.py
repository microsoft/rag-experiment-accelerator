import argparse
import mlflow
from azureml.pipeline import initialise_mlflow_client

from rag_experiment_accelerator.checkpoint import init_checkpoint
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import mlflow_run_name
from rag_experiment_accelerator.run.querying import run
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
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
    config = Config.from_path(
        environment,
        args.config_path,
    )
    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    handler = QueryOutputHandler(config.path.query_data_dir)
    init_checkpoint(config)

    for index_config in config.index.flatten():
        with mlflow.start_run(run_name=mlflow_run_name(config.job_name)):
            run(environment, config, index_config, mlflow_client)

            index_name = index_config.index_name()
            create_data_asset(
                data_path=handler.get_output_path(
                    index_name, config.experiment_name, config.job_name
                ),
                data_asset_name=index_name,
                environment=environment,
            )
