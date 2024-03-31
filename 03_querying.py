import argparse

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.run.querying import run
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)


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
    config = Config(
        environment,
        args.config_path,
    )

    handler = QueryOutputHandler(config.QUERY_DATA_LOCATION)
    for index_config in config.index_configs():
        run(environment, config, index_config)

        create_data_asset(
            data_path=handler.get_output_path(
                index_config.index_name(), config.EXPERIMENT_NAME, config.JOB_NAME
            ),
            data_asset_name=index_config.index_name(),
            environment=environment,
        )
