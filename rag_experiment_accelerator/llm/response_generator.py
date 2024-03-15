import logging

from openai import AzureOpenAI
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


class ResponseGenerator:
    def __init__(self, environment: Environment, config: Config, deployment_name: str):
        self.config = config
        self.deployment_name = deployment_name
        self.temperature = self.config.TEMPERATURE
        self.client = self._initialize_azure_openai_client(environment)

    def _initialize_azure_openai_client(self, environment: Environment):
        return AzureOpenAI(
            azure_endpoint=environment.openai_endpoint,
            api_key=environment.openai_api_key,
            api_version=environment.openai_api_version,
        )

    def generate_response(self, sys_message, prompt) -> str:
        """
        Generates a response to a given prompt using the OpenAI Chat API.

        Args:
            sys_message (str): The system message to include in the prompt.
            prompt (str): The user's prompt to generate a response to.

        Returns:
            str: The generated response to the user's prompt.
        """

        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": prompt},
        ]

        response = self._create_chat_completion_with_retry(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature,
        )

        # TODO: It is possible that this will return None.
        #       We need to ensure that this is handled properly in the places where this function gets called.
        if response.choices[0].finish_reason == "content_filter":
            logger.error(f"response not ideal {response.choices[0].finish_reason}")
            raise ContentFilteredException("Content was filtered.")
        return response.choices[0].message.content

    @retry(
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def _create_chat_completion_with_retry(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
