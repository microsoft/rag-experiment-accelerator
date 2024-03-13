import os
import sys
import argparse
import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.run.querying import run as query_run  # noqa: E402


def main():
    """Main function of the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument(
        "--eval_data_path", type=str, help="input: path to the data to evaluate on"
    )
    parser.add_argument(
        "--index_name_path",
        type=str,
        help="input: path to a file containing index name",
    )
    parser.add_argument(
        "--keyvault_name",
        type=str,
        help="input: keyvault name to load the environment from",
    )
    parser.add_argument(
        "--query_result_dir",
        type=str,
        help="output: path to write results of querying to",
    )
    args = parser.parse_args()

    environment = Environment.from_keyvault(args.keyvault_name)

    config = Config(environment, args.config_path)
    config.EVAL_DATA_JSONL_FILE_PATH = args.eval_data_path
    config.QUERY_DATA_LOCATION = args.query_result_dir

    with open(args.index_name_path, "r") as f:
        index_name = f.readline()
    index_config = IndexConfig.from_index_name(index_name, config)

    with mlflow.start_run():
        mlflow_client = mlflow.MlflowClient()
        active_run_data = mlflow_client.get_run(mlflow.active_run().info.run_id).data
        print(f"Active run data: {active_run_data}")
        parent_run_id = active_run_data.tags["azureml.pipeline"]
        print(f"Parent run ID: {parent_run_id}")

    query_run(environment, config, index_config)


if __name__ == "__main__":
    main()
