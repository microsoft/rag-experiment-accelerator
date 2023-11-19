import openai
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.utils import get_env_var, mask_string

logger = get_logger(__name__)

class OpenAICredentials:
    """
    A class to store OpenAI credentials.

    Attributes:
        OPENAI_API_TYPE (str): The type of OpenAI API to use.
        OPENAI_API_KEY (str): The API key for the OpenAI API.
        OPENAI_API_VERSION (str): The version of the OpenAI API to use.
        OPENAI_ENDPOINT (str): The endpoint for the OpenAI API.

    Methods:
        __init__(self, openai_api_type: str, openai_api_key: str, openai_api_version: str, openai_endpoint: str) -> None:
            Initializes the OpenAICredentials object.
        from_env(cls) -> "OpenAICredentials":
            Creates an OpenAICredentials object from environment variables.
        _set_credentials(self) -> None:
            Sets the OpenAI credentials.
    """

    def __init__(
        self,
        openai_api_type: str,
        openai_api_key: str,
        openai_api_version: str,
        openai_endpoint: str,
    ) -> None:
        """
        Initializes the OpenAICredentials object.

        Args:
            openai_api_type (str): The type of OpenAI API to use.
            openai_api_key (str): The API key for the OpenAI API.
            openai_api_version (str): The version of the OpenAI API to use.
            openai_endpoint (str): The endpoint for the OpenAI API.

        Raises:
            ValueError: If openai_api_type is not 'azure' or 'open_ai'.
        """
        if openai_api_type is not None and openai_api_type not in ["azure", "open_ai"]:
            logger.critical("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")
            raise ValueError("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")

        self.OPENAI_API_TYPE = openai_api_type
        self.OPENAI_API_KEY = openai_api_key
        self.OPENAI_API_VERSION = openai_api_version
        self.OPENAI_ENDPOINT = openai_endpoint

        self.set_credentials()

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        """
        Creates an OpenAICredentials object from environment variables.

        Returns:
            OpenAICredentials: The OpenAICredentials object.
        """
        return cls(
            openai_api_type=get_env_var(
                var_name="OPENAI_API_TYPE",
                critical=False,
                mask=False,
            ),
            openai_api_key=get_env_var(
                var_name="OPENAI_API_KEY", critical=False, mask=True
            ),
            openai_api_version=get_env_var(
                var_name="OPENAI_API_VERSION",
                critical=False,
                mask=False,
            ),
            openai_endpoint=get_env_var(
                var_name="OPENAI_ENDPOINT",
                critical=False,
                mask=True,
            ),
        )

    def set_credentials(self) -> None:
        """
        Sets the OpenAI credentials.
        """
        openai.api_type = self.OPENAI_API_TYPE
        openai.api_key = self.OPENAI_API_KEY
        logger.info(f"OpenAI API key set to {mask_string(openai.api_key)}")
        if self.OPENAI_API_TYPE == "open_ai":
                openai.api_version = None
                openai.api_base = None
        elif self.OPENAI_API_TYPE == "azure":
            openai.api_version = self.OPENAI_API_VERSION
            openai.api_base = self.OPENAI_ENDPOINT