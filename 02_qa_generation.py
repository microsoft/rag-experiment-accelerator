from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.qa_generation import run

if __name__ == "__main__":
    directory = ArgumentParser().get_directory_arg()
    filename = ArgumentParser().get_config_filename_arg()
    data_directory = ArgumentParser().get_data_directory_arg()
    run(directory, data_directory, filename=filename)
