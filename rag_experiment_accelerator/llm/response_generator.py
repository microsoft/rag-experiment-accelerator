import logging

import json
import openai

from openai import AzureOpenAI
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type,
)

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.llm.prompt.prompt import (
    StructuredPrompt,
    CoTPrompt,
    Prompt,
    PromptTag,
)

logger = get_logger(__name__)


class ResponseGenerator:
    def __init__(self, environment: Environment, config: Config, deployment_name: str):
        self.config = config
        self.deployment_name = deployment_name
        self.temperature = self.config.TEMPERATURE
        self.use_long_prompt = True
        self.client = self._initialize_azure_openai_client(environment)
        self.json_object_supported = True

    def _initialize_azure_openai_client(self, environment: Environment):
        return AzureOpenAI(
            azure_endpoint=environment.openai_endpoint,
            api_key=environment.openai_api_key,
            api_version=environment.openai_api_version,
        )

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

    @retry(
        before_sleep=before_sleep_log(logger, logging.CRITICAL),
        after=after_log(logger, logging.CRITICAL),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(
            (ContentFilteredException, TypeError, KeyboardInterrupt)
        ),
    )
    def _get_response(
        self, messages, prompt: Prompt, temperature: float | None = None
    ) -> any:
        kwargs = {}

        if self.json_object_supported and PromptTag.JSON in prompt.tags:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                **kwargs,
            )
        except openai.BadRequestError as e:
            if e.param == "response_format":
                self.json_object_supported = False
            raise e

        if response.choices[0].finish_reason == "content_filter":
            logger.error(
                f"Response was filtered {response.choices[0].finish_reason}:\n{response}"
            )
            raise ContentFilteredException("Content was filtered.")

        response_text = response.choices[0].message.content

        return self._interpret_response(response_text, prompt)

    def generate_response(
        self, prompt: Prompt, temperature: float | None = None, **kwargs
    ) -> any:
        system_arguments = Prompt.arguments_in_prompt(prompt.system_message)
        user_arguments = Prompt.arguments_in_prompt(prompt.user_template)

        for key in system_arguments:
            assert key in kwargs, f"Missing argument {key} in system message."
        for key in user_arguments:
            assert key in kwargs, f"Missing argument {key} in user template."

        sys_message = prompt.system_message.format(
            **{key: value for key, value in kwargs.items() if key in system_arguments}
        )
        user_template = prompt.user_template.format(
            **{key: value for key, value in kwargs.items() if key in user_arguments}
        )

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
