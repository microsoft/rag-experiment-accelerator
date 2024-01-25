from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.evaluation import run

if __name__ == "__main__":
    directory = ArgumentParser().get_directory_arg()
    filename = ArgumentParser().get_config_filename_arg()
    run(directory, filename)
