import json
import argparse
import mlflow

from azureml.pipeline import initialise_mlflow_client

from rag_experiment_accelerator.checkpoint import init_checkpoint
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import get_all_file_paths, mlflow_run_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument("--data_dir", type=str, help="input: path to the input data")
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    config = Config.from_path(environment, args.config_path, args.data_dir)
    init_checkpoint(config)
    file_paths = get_all_file_paths(config.path.data_dir)
    mlflow_client = initialise_mlflow_client(environment, config)
    mlflow.set_experiment(config.experiment_name)

    index_dict = {"indexes": []}

    for index_config in config.index.flatten():
        with mlflow.start_run(run_name=mlflow_run_name(f"index_job_{config.job_name}")):
            index_name = run(
                environment, config, index_config, file_paths, mlflow_client
            )
            index_dict["indexes"].append(index_name)

    # saves the list of index names locally, not used afterwards
    with open(config.path.generated_index_names_file, "w") as index_names_file:
        json.dump(index_dict, index_names_file, indent=4)
