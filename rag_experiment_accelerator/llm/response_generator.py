from openai import AzureOpenAI
from rag_experiment_accelerator.config.config import Config


class ResponseGenerator:

    def __init__(self):
        self.config = Config()
        self.client = self._initialize_azure_openai_client()

    def _initialize_azure_openai_client(self):
            return AzureOpenAI(
                azure_endpoint=self.config.OpenAICredentials.OPENAI_ENDPOINT,
                api_key=self.config.OpenAICredentials.OPENAI_API_KEY,
                api_version=self.config.OpenAICredentials.OPENAI_API_VERSION
            )


    def generate_response(self, sys_message, prompt, aoai_deployment_name, temperature):
        """
        Generates a response to a given prompt using the OpenAI Chat API.

        Args:
            sys_message (str): The system message to include in the prompt.
            prompt (str): The user's prompt to generate a response to.
            aoai_deployment_name (str): The name of the Azure OpenAI deployment to use for generating the response.
            temperature (float): Controls the "creativity" of the response. Higher values result in more creative responses.

        Returns:
            str: The generated response to the user's prompt.
        """

        messages = [
                {"role": "system", "content": sys_message}, 
                {"role": "user", "content": prompt}
            ]

        response = self.client.chat.completions.create(
            model=aoai_deployment_name, 
            messages=messages,
            temperature=temperature
        )

        # TODO: It is possible that this will return None. 
        #       We need to ensure that this is handled properly in the places where this function gets called.
        return response.choices[0].message.content