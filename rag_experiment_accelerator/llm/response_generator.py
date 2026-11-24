import json

from string import Template
import abc

from rag_experiment_accelerator.config.llm_config import BaseLLMConfig
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.llm.prompt.prompt import (
    StructuredPrompt,
    CoTPrompt,
    Prompt,
    PromptTag,
)

logger = get_logger(__name__)


class ResponseGenerator:
    def __init__(self, config: BaseLLMConfig, **kwargs):
        self.config = config
        self.use_long_prompt = True
        self.json_object_supported = True

    def _interpret_response(self, response: str, prompt: Prompt) -> any:
        interpreted_response = response

        if PromptTag.ChainOfThought in prompt.tags:
            if not isinstance(prompt, CoTPrompt):
                raise TypeError(
                    "Prompt is not a CoTPrompt but has Chain-of-thought tag"
                )

            splitted = interpreted_response.split(prompt.separator)
            assert len(splitted) != 1, f"Separator not found in response: {response}"
            assert (
                len(splitted) <= 2
            ), f"More than one separator found in response: {response}"
            interpreted_response = splitted[1]

        if PromptTag.Structured in prompt.tags:
            if not isinstance(prompt, StructuredPrompt):
                raise TypeError(
                    "Prompt is not a StructuredPrompt but has Structured tag"
                )
            assert prompt.validator(
                interpreted_response
            ), f"Response {response} does not match the expected format"

        if PromptTag.JSON in prompt.tags:
            interpreted_response = json.loads(interpreted_response)

        return interpreted_response

    @abc.abstractmethod
    def _get_response(self, messages, prompt: Prompt, temperature: float) -> any:
        raise NotImplementedError

    def generate_response(
        self,
        prompt: Prompt,
        temperature: float | None = None,
        prompt_last: bool = False,
        **kwargs,
    ) -> any:
        system_arguments = Prompt.arguments_in_prompt(prompt.system_message)
        user_arguments = Prompt.arguments_in_prompt(prompt.user_template)

        if temperature is None:
            temperature = self.config.temperature

        for key in system_arguments:
            assert key in kwargs, f"Missing argument {key} in system message."

        for key in user_arguments:
            assert key in kwargs, f"Missing argument {key} in user template."

        sys_template = Template(prompt.system_message)
        sys_message = sys_template.safe_substitute(
            **{key: value for key, value in kwargs.items() if key in system_arguments}
        )

        user_template = Template(prompt.user_template)
        user_template = user_template.safe_substitute(
            **{key: value for key, value in kwargs.items() if key in user_arguments}
        )

        if prompt_last:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{user_template}\n{sys_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": user_template},
            ]

        try:
            response = self._get_response(messages, prompt, temperature)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if PromptTag.NonStrict in prompt.tags:
                logger.debug(f"Failed to generate response: {e}")
                return None
            else:
                raise e

        return response
