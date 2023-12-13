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

def generate_response(sys_message, prompt, engine_model, temperature):
    """
    Generates a response to a given prompt using the OpenAI Chat API.

    Args:
        sys_message (str): The system message to include in the prompt.
        prompt (str): The user's prompt to generate a response to.
        engine_model (str): The name of the OpenAI engine model to use for generating the response.
        temperature (float): Controls the "creativity" of the response. Higher values result in more creative responses.

    Returns:
        str: The generated response to the user's prompt.
    """
    config = Config()
    

    messages = [
            {"role": "system", "content": sys_message}, 
            {"role": "user", "content": prompt}
        ]
    if config.OpenAICredentials.OPENAI_API_TYPE == 'azure':
        client = AzureOpenAI(
            azure_endpoint=config.OpenAICredentials.OPENAI_ENDPOINT, 
            api_key=config.OpenAICredentials.OPENAI_API_KEY,  
            api_version=config.OpenAICredentials.OPENAI_API_VERSION
        )
    else:
        client = OpenAI(
            api_key=config.OpenAICredentials.OPENAI_API_KEY,  
        )

    response = create_chat_completion(
        client,
        model=engine_model,  # model = "deployment_name" for AzureOpenAI
        messages=messages,
        temperature=temperature
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
def create_chat_completion(client, **kwargs):
    return client.chat.completions.create(**kwargs)
