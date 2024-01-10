import argparse
import os


class ArgumentParser:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ArgumentParser, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.parser = argparse.ArgumentParser("rag_experiment_accelerator")
        self.parser.add_argument(
            "-d",
            "--directory",
            help=(
                "The directory holding the configuration files and data."
                " Defaults to current working directory"
            ),
            type=str,
            default=os.getcwd(),
        )
        self.parser.add_argument(
            "-dd",
            "--data-directory-name",
            help=("The directory holding the data. Defaults to 'data'"),
            type=str,
            default="data",
        )
        self.parser.add_argument(
            "-cf",
            "--config-file-name",
            help=("JSON config file that. Defaults to 'config.json'"),
            type=str,
            default="config.json",
        )

    def get_directory_arg(self):
        return self.parser.parse_args().directory

    def get_data_directory_arg(self):
        return self.parser.parse_args().data_directory_name

    def get_config_directory_arg(self):
        return self.parser.parse_args().config_file_name
