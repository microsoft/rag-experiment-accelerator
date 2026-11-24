from rag_experiment_accelerator.config.llm_config import BaseLLMConfig
from rag_experiment_accelerator.config.environment import Environment

from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.llm.huggingface_response_generator import (
    HuggingfaceResponseGenerator,
)
from rag_experiment_accelerator.llm.openai_response_generator import (
    OpenAIResponseGenerator,
)


def get_response_generator(
    config: BaseLLMConfig, environment: Environment
) -> ResponseGenerator:
    if config.llm_type == "openai":
        return OpenAIResponseGenerator(config, environment)
    elif config.llm_type == "huggingface":
        return HuggingfaceResponseGenerator(config)
    else:
        raise ValueError(f"Unsupported LLM type: {config.llm_type}")
