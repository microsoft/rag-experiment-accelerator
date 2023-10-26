import os
import openai
from dotenv import load_dotenv

load_dotenv()


def init_openai():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_ENDPOINT')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
