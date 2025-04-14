from dataclasses import dataclass, field
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class BaseLLMConfig(BaseConfig):
    llm_type: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 100


@dataclass
class LLMConfig(BaseConfig):
    chat_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    eval_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
