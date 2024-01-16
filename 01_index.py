from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.index import run

if __name__ == "__main__":
    config_directory = ArgumentParser().get_directory_arg()
    data_directory = ArgumentParser().get_data_directory_arg()
    run(config_directory, data_directory)
