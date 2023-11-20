import openai
from rag_experiment_accelerator.llm.base import LLMModel
from rag_experiment_accelerator.config.auth import OpenAICredentials

from rag_experiment_accelerator.utils.logging import get_logger
logger = get_logger(__name__)

class OpenAIModel(LLMModel):

    def __init__(self, model_name: str, tags: list[str], openai_creds: OpenAICredentials, *args, **kwargs) -> None:
        super().__init__(model_name=model_name, *args, **kwargs)
        self._openai_creds = openai_creds
        self._tags = tags


    def try_retrieve_model(self):
        try:
            model = openai.Model.retrieve(self.model_name)
            if self._openai_creds.OPENAI_API_TYPE == "open_ai":
                return model
            
            if model["status"] != "succeeded":
                logger.critical(f"Model {self.model_name} is not ready.")
                raise ValueError(f"Model {self.model_name} is not ready.")
            
            for tag in self._tags:
                if not model["capabilities"][tag]:
                    logger.critical(
                        f"Model {self.model_name} does not have the {tag} capability."
                    )
                    raise ValueError(
                        f"Model {self.model_name} does not have the {tag} capability."
                    )
            return model
        except openai.error.InvalidRequestError:
            logger.critical(f"Model {self.model_name} does not exist.")
            raise ValueError(f"Model {self.model_name} does not exist.")


# class OpenAIChatCompletionModel(OpenAIModel):

#     def __init__(self, model_name: str, openai_creds: OpenAICredentials, temperature: float) -> None:
#         super().__init__(model_name=model_name, tags=["chat_completion", "inference"])
#         self._temperature = temperature
#         self._openai_creds = openai_creds
    
#     def generate_response(self, sys_message, prompt):
#         """
#         Generates a response to a given prompt using the OpenAI Chat API.

#         Args:
#             sys_message (str): The system message to include in the prompt.
#             prompt (str): The user's prompt to generate a response to.
#             engine_model (str): The name of the OpenAI engine model to use for generating the response.
#             temperature (float): Controls the "creativity" of the response. Higher values result in more creative responses.

#         Returns:
#             str: The generated response to the user's prompt.
#         """
#         prompt_structure = [
#             {
#                 "role": "system",
#                 "content": sys_message,
#             },
#             {
#                 "role": "user", 
#                 "content": prompt
#             }
#         ]

#         params = {
#             "messages": prompt_structure,
#             "temperature": self._temperature,
#         }
#         if self._openai_creds.OPENAI_API_TYPE == "azure":
#             params["engine"] = self.model_name
#         else:
#             params["model"] = self.model_name

#         self._openai_creds.set_credentials()
#         response = openai.ChatCompletion.create(**params)
#         answer = response.choices[0]["message"]["content"]
#         return answer


