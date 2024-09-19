from rag_experiment_accelerator.checkpoint import init_checkpoint
import os
import sys
import argparse
from typing import List

import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.run.index import run as index_run  # noqa: E402


def init():
    """Main function of the script."""

    global args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument("--data_dir", type=str, help="input: path to the data")
    parser.add_argument("--index_name", type=str, help="input: experiment index name")
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
        "--index_name_path",
        type=str,
        help="output: path to write a file with index name",
    )

    args, _ = parser.parse_known_args()

    global config
    global environment
    global index_config
    global mlflow_client

    environment = Environment.from_keyvault(args.keyvault)
    config = Config.from_path(environment, args.config_path, args.data_dir)
    init_checkpoint(config)

    index_config = IndexConfig.from_index_name(args.index_name)
    mlflow_client = mlflow.MlflowClient(args.mlflow_tracking_uri)


def run(input_paths: List[str]) -> str:
    global args
    global config
    global environment
    global index_config
    global mlflow_client

    index_run(environment, config, index_config, input_paths, mlflow_client)

    return [args.index_name]
