import openai  
import os
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_type = "azure"
openai.api_base = os.environ['OPENAI_ENDPOINT']
openai.api_version = os.environ['OPENAI_API_VERSION'] # "2023-03-15-preview"


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
    prompt_structure = [{
        'role': 'system',
        'content': sys_message,
    }]
    
    prompt_structure.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(engine=engine_model, messages=prompt_structure, temperature=temperature)
    answer = response.choices[0]['message']['content']
    return answer