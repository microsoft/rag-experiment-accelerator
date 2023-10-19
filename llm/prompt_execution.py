import openai  
import os

# Added as ENV variables not available for local execution #37
from dotenv import load_dotenv
# Added as ENV variables not available for local execution #37
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_type = "azure"
openai.api_base = os.environ['OPENAI_ENDPOINT']
openai.api_version = os.environ['OPENAI_API_VERSION'] # "2023-03-15-preview"


def generate_response(sys_message,prompt,engine_model, temperature):
    prompt_structure = [{
        'role': 'system',
        'content': sys_message,
    }]
    
    prompt_structure.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(engine=engine_model, messages=prompt_structure, temperature=temperature)
    answer = response.choices[0]['message']['content']
    return answer