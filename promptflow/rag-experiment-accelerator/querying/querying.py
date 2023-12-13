from promptflow import tool
from rag_experiment_accelerator.run.querying import run


@tool
def my_python_tool(config_dir: str) -> bool:
    try:
        run(config_dir)
    except Exception as e:
        print(e)
        
        return False	
    return True