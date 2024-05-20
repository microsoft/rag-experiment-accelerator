import json
import argparse
import mlflow

from azureml.pipeline import initialise_mlflow_client

from rag_experiment_accelerator.checkpoint import CheckpointFactory
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import (
    formatted_datetime_suffix,
    get_all_file_paths,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument("--data_dir", type=str, help="input: path to the input data")
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    config = Config(environment, args.config_path, args.data_dir)
    CheckpointFactory.create_checkpoint(
        config.execution_environment, config.use_checkpoints, config.artifacts_dir
    )
    file_paths = get_all_file_paths(config.data_dir)
    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    for index_config in config.index_configs():
        with mlflow.start_run(
            run_name=f"index_job_{config.job_name}_{formatted_datetime_suffix()}"
        ):
            index_dict = run(
                environment, config, index_config, file_paths, mlflow_client
            )

    with open(config.generated_index_names_file_path, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)
