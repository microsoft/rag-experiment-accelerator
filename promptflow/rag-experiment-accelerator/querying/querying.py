from promptflow import tool
from rag_experiment_accelerator.run.querying import run


@tool
def my_python_tool(search_config_dir: str) -> bool:
    try:
        run(search_config_dir)
    except Exception as e:
        print(e)
        return False
    return True
