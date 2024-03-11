import os
import sys
import argparse
from typing import List

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.run.index import run as index_run  # noqa: E402


def init():
    """Main function of the script."""

    global args

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, help="directory with input configs")
    parser.add_argument("--data_dir", type=str, help="directory with input data")
    parser.add_argument(
        "--config_path", type=str, help="relative path to the config file"
    )
    parser.add_argument("--index_name", type=str, help="experiment index name")
    parser.add_argument(
        "--keyvault_name", type=str, help="keyvault name to load the environment from"
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

    environment = Environment.from_keyvault(args.keyvault_name)
    config = Config(environment, args.config_dir, args.data_dir, args.config_path)
    config.data_dir = args.data_dir
    index_config = IndexConfig.from_index_name(args.index_name, config)


def run(input_paths: List[str]) -> List[str]:
    global args
    global config
    global environment
    global index_config

    index_run(environment, config, index_config, input_paths)

    return [args.index_name]
