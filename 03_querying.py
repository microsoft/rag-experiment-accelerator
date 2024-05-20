import argparse
import mlflow
from azureml.pipeline import initialise_mlflow_client

from rag_experiment_accelerator.checkpoint import CheckpointFactory
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import formatted_datetime_suffix
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
        default=None,  # default is initialised in Config
    )
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    config = Config(
        environment,
        args.config_path,
    )
    CheckpointFactory.create_checkpoint(
        config.execution_environment, config.use_checkpoints, config.artifacts_dir
    )

    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    handler = QueryOutputHandler(config.query_data_location)

    for index_config in config.index_configs():
        with mlflow.start_run(
            run_name=f"query_job_{config.job_name}_{formatted_datetime_suffix()}"
        ):
            run(environment, config, index_config, mlflow_client)

            create_data_asset(
                data_path=handler.get_output_path(
                    index_config.index_name(), config.experiment_name, config.job_name
                ),
                data_asset_name=index_config.index_name(),
                environment=environment,
            )
