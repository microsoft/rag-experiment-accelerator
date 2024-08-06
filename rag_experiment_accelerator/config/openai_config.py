from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class OpenAIConfig(BaseConfig):
    azure_oai_chat_deployment_name: str = "gpt-35-turbo"
    azure_oai_eval_deployment_name: str = "gpt-35-turbo"
    openai_temperature: float = 0.0
