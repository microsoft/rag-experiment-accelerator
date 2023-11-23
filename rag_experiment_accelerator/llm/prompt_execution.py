import openai
import time
from openai.error import RateLimitError
from rag_experiment_accelerator.utils.logging import get_logger


retry_count = 5
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
    prompt_structure = [
        {
            "role": "system",
            "content": sys_message,
        }
    ]

    prompt_structure.append({"role": "user", "content": prompt})

    params = {
        "messages": prompt_structure,
        "temperature": temperature,
    }
    if openai.api_type == "azure":
        params["engine"] = engine_model
    else:
        params["model"] = engine_model

    for i in range(retry_count):
        try:
            response = openai.ChatCompletion.create(**params)
            return response.choices[0]["message"]["content"]
        except RateLimitError as e:
            logger.warning("Recieved rate limit error. Retrying in 10 seconds...", e)
            time.sleep(10)

    raise Exception("Maximum retries reached", e)
