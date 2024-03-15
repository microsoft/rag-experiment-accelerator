import json
import argparse

from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import get_all_file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument("--data_dir", type=str, help="input: path to the input data")
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    config = Config(environment, args.config_path, args.data_dir)

    file_paths = get_all_file_paths(config.data_dir)
    for index_config in config.index_configs():
        index_dict = run(environment, config, index_config, file_paths)

    with open(config.GENERATED_INDEX_NAMES_FILE_PATH, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)
