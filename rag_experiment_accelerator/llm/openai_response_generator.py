import logging

import openai

from rag_experiment_accelerator.llm.response_generator import ResponseGenerator

from openai import AzureOpenAI
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type,
)

from rag_experiment_accelerator.config.llm_config import BaseLLMConfig
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.llm.prompt.prompt import (
    Prompt,
    PromptTag,
)

logger = get_logger(__name__)


class OpenAIResponseGenerator(ResponseGenerator):
    def __init__(self, config: BaseLLMConfig, environment: Environment):
        super().__init__(config)
        self.client = self._initialize_azure_openai_client(environment)

    def _initialize_azure_openai_client(self, environment: Environment):
        return AzureOpenAI(
            azure_endpoint=environment.openai_endpoint,
            api_key=environment.openai_api_key,
            api_version=environment.openai_api_version,
        )

    @retry(
        before_sleep=before_sleep_log(logger, logging.CRITICAL),
        after=after_log(logger, logging.CRITICAL),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(
            (ContentFilteredException, TypeError, KeyboardInterrupt)
        ),
    )
    def _get_response(self, messages, prompt: Prompt, temperature: float) -> any:
        kwargs = {}

        if self.json_object_supported and PromptTag.JSON in prompt.tags:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                **kwargs,
            )
        except openai.BadRequestError as e:
            if e.param == "response_format":
                self.json_object_supported = False
                return self._get_response(messages, prompt, temperature)
            raise e

        if response.choices[0].finish_reason == "content_filter":
            logger.error(
                f"Response was filtered {response.choices[0].finish_reason}:\n{response}"
            )
            raise ContentFilteredException("Content was filtered.")

        response_text = response.choices[0].message.content

        return self._interpret_response(response_text, prompt)
