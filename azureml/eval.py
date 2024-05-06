import os
import sys
import argparse
import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.run.evaluation import run as eval_run  # noqa: E402


def main():
    """Main function of the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument(
        "--index_name_path",
        type=str,
        help="input: path to a file containing index name",
    )
    parser.add_argument(
        "--query_result_dir",
        type=str,
        help="input: path to read results of querying from",
    )
    parser.add_argument(
        "--keyvault", type=str, help="keyvault to load the environment from"
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
        "--eval_result_dir",
        type=str,
        help="output: path to write results of evaluation to",
    )
    args = parser.parse_args()

    environment = Environment.from_keyvault(args.keyvault)
    config = Config(environment, config_path=args.config_path)
    with open(args.index_name_path, "r") as f:
        index_name = f.readline()
    index_config = IndexConfig.from_index_name(index_name, config)

    config.query_data_location = args.query_result_dir
    config.eval_data_location = args.eval_result_dir
    mlflow_client = mlflow.MlflowClient(args.mlflow_tracking_uri)
    eval_run(environment, config, index_config, mlflow_client, name_suffix="_result")


if __name__ == "__main__":
    main()
