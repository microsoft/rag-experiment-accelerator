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
