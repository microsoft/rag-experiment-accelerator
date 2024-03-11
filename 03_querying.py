from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.querying import run
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    environment = Environment.from_env()
    config = Config(
        environment,
        arg_parser.get_directory_arg(),
        arg_parser.get_data_directory_arg(),
        arg_parser.get_config_filename_arg(),
    )

    handler = QueryOutputHandler(config.QUERY_DATA_LOCATION)
    for index_config in config.index_configs():
        run(environment, config, index_config)

        create_data_asset(
            data_path=handler.get_output_path(index_config.index_name()),
            data_asset_name=index_config.index_name(),
            environment=environment,
        )
