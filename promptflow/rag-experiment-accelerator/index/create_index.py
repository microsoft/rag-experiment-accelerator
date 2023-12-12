import os

from promptflow import tool
from promptflow.connections import CustomConnection
from rag_experiment_accelerator.run.index import run

@tool
def my_python_tool(should_index: bool, config_dir: str, connection: CustomConnection) -> bool:
    if should_index:
        try:
            os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"] = connection.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
            os.environ["AZURE_SEARCH_ADMIN_KEY"] = connection.secrets["AZURE_SEARCH_ADMIN_KEY"]
            os.environ["OPENAI_API_KEY"] = connection.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_TYPE"] = connection.secrets["OPENAI_API_TYPE"]
            os.environ["OPENAI_ENDPOINT"] = connection.secrets["OPENAI_ENDPOINT"]
            os.environ["OPENAI_API_VERSION"] = connection.secrets["OPENAI_API_VERSION"]
            os.environ["AML_SUBSCRIPTION_ID"] = connection.secrets["AML_SUBSCRIPTION_ID"]
            os.environ["AML_RESOURCE_GROUP_NAME"] = connection.secrets["AML_RESOURCE_GROUP_NAME"]
            os.environ["AML_WORKSPACE_NAME"] = connection.secrets["AML_WORKSPACE_NAME"]
            run(config_dir)
        except Exception as e:
            print(e)
