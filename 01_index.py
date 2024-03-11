import json

from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.index import run
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import get_all_files

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    environment = Environment.from_env()
    config = Config(
        environment,
        arg_parser.get_directory_arg(),
        arg_parser.get_data_directory_arg(),
        arg_parser.get_config_filename_arg(),
    )

    file_paths = get_all_files(config.data_dir)
    for index_config in config.index_configs():
        index_dict = run(environment, config, index_config, file_paths)

    with open(config.GENERATED_INDEX_NAMES_FILE_PATH, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)
