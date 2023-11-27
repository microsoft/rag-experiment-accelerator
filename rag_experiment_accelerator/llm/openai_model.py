import openai
from rag_experiment_accelerator.llm.base import LLMModel
from rag_experiment_accelerator.config.config import OpenAICredentials

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIModel(LLMModel):
    """
    A class representing an OpenAI model.
    Args:
        model_name (str): The name of the model.
        tags (list[str]): A list of tags associated with the model.
        openai_creds (OpenAICredentials): An instance of the OpenAICredentials class.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    Attributes:
        _openai_creds (OpenAICredentials): An instance of the OpenAICredentials class.
        _tags (list[str]): A list of tags associated with the model.
    Methods:
        try_retrieve_model: Tries to retrieve the model and performs necessary checks.
    """

    def __init__(
        self,
        model_name: str,
        tags: list[str],
        openai_creds: OpenAICredentials,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name, *args, **kwargs)
        self._openai_creds = openai_creds
        self._tags = tags

    def try_retrieve_model(self):
        """
        Tries to retrieve the model and performs necessary checks.
        Returns:
            model: The retrieved model.
        Raises:
            ValueError: If the model is not ready or does not have the required capabilities.
        """
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
