import os
import sys
import argparse

import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.checkpoint import init_checkpoint  # noqa: E402
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
        "--keyvault",
        type=str,
        help="input: keyvault to load the environment from",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        help="input: mlflow tracking uri to log to",
    )
    parser.add_argument(
        "--mlflow_parent_run_id",
        type=str,
        help="input: mlflow parent run id to connect nested run to",
    )
    parser.add_argument(
        "--query_result_dir",
        type=str,
        help="output: path to write results of querying to",
    )
    args = parser.parse_args()

    environment = Environment.from_keyvault(args.keyvault)

    config = Config(environment, args.config_path)
    config.path.eval_data_file = args.eval_data_path
    config.path.query_data_dir = args.query_result_dir
    init_checkpoint(config)

    with open(args.index_name_path, "r") as f:
        index_name = f.readline()
    index_config = IndexConfig.from_index_name(index_name)

    mlflow_client = mlflow.MlflowClient(args.mlflow_tracking_uri)
    query_run(environment, config, index_config, mlflow_client)


if __name__ == "__main__":
    main()
