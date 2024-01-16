from rag_experiment_accelerator.run.argument_parser import ArgumentParser
from rag_experiment_accelerator.run.qa_generation import run

if __name__ == "__main__":
    directory = ArgumentParser().get_directory_arg()
    run(directory)
