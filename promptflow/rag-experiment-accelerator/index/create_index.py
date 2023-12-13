from promptflow import tool
from rag_experiment_accelerator.run.index import run

@tool
def my_python_tool(should_index: bool, config_dir: str) -> bool:
    if should_index:
        try:
            run(config_dir)
        except Exception as e:
            print(e)
