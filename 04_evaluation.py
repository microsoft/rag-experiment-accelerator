import argparse

import mlflow

from azureml.pipeline import initialise_mlflow_client
from rag_experiment_accelerator.checkpoint import create_checkpoint
from rag_experiment_accelerator.run.evaluation import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import formatted_datetime_suffix


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
    config = Config(environment, args.config_path, args.data_dir)
    create_checkpoint(
        config.execution_environment, config.use_checkpoints, config.artifacts_dir
    )

    name_suffix = formatted_datetime_suffix()
    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    for index_config in config.index_configs():
        with mlflow.start_run(
            run_name=f"eval_job_{config.job_name}_{formatted_datetime_suffix()}"
        ):
            run(
                environment,
                config,
                index_config,
                mlflow_client,
                name_suffix=name_suffix,
            )
