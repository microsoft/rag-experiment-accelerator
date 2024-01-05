
import argparse
import os


def get_directory_arg():
    parser = argparse.ArgumentParser("rag_experiment_accelerator")
    parser.add_argument("-d", "--directory", help="The directory holding the configuration files and data. Config file must be named config.json and data must but under <DIRECTORY>/data. Defaults to current working directory", type=str, default=os.getcwd())
    args = parser.parse_args()
    return args.directory
