import logging
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import AzureOpenAI, OpenAI
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class ResponseGenerator:

    def __init__(self, deployment_name):
        self.config = Config()
        self.deployment_name = deployment_name
        self.temperature = self.config.TEMPERATURE
        self.client = self._initialize_azure_openai_client()

    def _initialize_azure_openai_client(self):

        return AzureOpenAI(
            azure_endpoint=self.config.OpenAICredentials.OPENAI_ENDPOINT,
            api_key=self.config.OpenAICredentials.OPENAI_API_KEY,
            api_version=self.config.OpenAICredentials.OPENAI_API_VERSION
        )

    def generate_response(self, sys_message, prompt):
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
            {"role": "user", "content": prompt}
        ]

        print(self.deployment_name,
              self.temperature)

        response = self._create_chat_completion_with_retry(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature
        )

        # TODO: It is possible that this will return None.
        #       We need to ensure that this is handled properly in the places where this function gets called.
        return response.choices[0].message.content

    @retry(
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    def _create_chat_completion_with_retry(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
