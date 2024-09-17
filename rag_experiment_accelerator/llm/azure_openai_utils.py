from openai import AzureOpenAI
from promptflow.core import AzureOpenAIModelConfiguration
from rag_experiment_accelerator.config.environment import Environment


def initialize_azure_openai_client(environment: Environment):
    return AzureOpenAI(
        azure_endpoint=environment.openai_endpoint,
        api_key=environment.openai_api_key,
        api_version=environment.openai_api_version,
    )


def initialize_azure_openai_model_config(environment: Environment,
                                         aoai_deployment_name: str):
    return AzureOpenAIModelConfiguration(
        azure_endpoint=environment.openai_endpoint,
        api_key=environment.openai_api_key,
        azure_deployment=aoai_deployment_name
    )
