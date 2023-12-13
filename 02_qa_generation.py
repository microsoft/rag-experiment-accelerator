from rag_experiment_accelerator.run.args import get_directory_arg
from rag_experiment_accelerator.run.qa_generation import run

if __name__ == "__main__":
    directory = get_directory_arg()
    run(directory)
