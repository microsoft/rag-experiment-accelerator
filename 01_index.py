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

    config = Config.from_path(environment, args.config_path, args.data_dir)

    file_paths = get_all_file_paths(config.path.data_dir)

    index_dict = {"indexes": []}

    for index_config in config.index_config.flatten():
        index_name = run(environment, config, index_config, file_paths)
        index_dict["indexes"].append(index_name)

    # saves the list of index names locally, not used afterwards
    with open(config.path.generated_index_names_file, "w") as index_names_file:
        json.dump(index_dict, index_names_file, indent=4)
