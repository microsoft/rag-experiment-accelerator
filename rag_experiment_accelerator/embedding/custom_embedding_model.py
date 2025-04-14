import urllib.request
import json
import os
import ssl
from typing import Union

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class CustomEmbeddingModel(EmbeddingModel):
    """
    A class representing a Custom Embedding Model deployed as an AzureML online endpoint.

    Args:
        model_name (str): The name of the deployment.
        environment (Environment): The initialized environment.
        dimension (int, optional): The dimension of the embedding. Defaults to 1536.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self, model_name: str, environment: Environment, dimension: int = 1536, **kwargs
    ):
        super().__init__(name=model_name, dimension=dimension, **kwargs)
        self.environment = environment
        pass

    def prepare_request(self, body: Union[dict, list]) -> Union[dict, bytes]:
        """
        Prepares the request to be sent to the AzureML online endpoint.

        Args:
            body (Union[dict, list]): The input data.

        Returns:
            Union[dict, bytes]: The prepared request body.

        """
        # replace the format based the model input
        data_format = {
            "input": body,
        }

        body = str.encode(json.dumps(data_format))

        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.environment.azure_model_api_key),
            "azureml-model-deployment": self.name,
        }

        return headers, body

    def make_request(self, body: bytes, headers: dict) -> list[float]:
        """
        Makes a request to the AzureML online endpoint.

        Args:
            body (bytes): The request body.
            headers (dict): The request headers.

        Returns:
            list[float]: The response from the AzureML online endpoint.

        """
        try:
            logger.info("Calling Custom Embedding Model API")
            req = urllib.request.Request(
                self.environment.azure_model_api_endpoint, body, headers
            )
            response = urllib.request.urlopen(req)
            logger.info("Custom Embedding Model response received")
            data = json.loads(response.read())
            logger.info("Custom Embedding Model response parsed")

            return data

        except urllib.error.HTTPError as error:
            logger.exception("The request failed with status code: " + str(error.code))
            raise

    def allowSelfSignedHttps(self, allowed: bool) -> None:
        """
        Allows self-signed HTTPS requests.

        Args:
            allowed (bool): Whether to allow self-signed HTTPS requests.

        """

        # bypass the server certificate verification on client side
        if (
            allowed
            and not os.environ.get("PYTHONHTTPSVERIFY", "")
            and getattr(ssl, "_create_unverified_context", None)
        ):
            ssl._create_default_https_context = ssl._create_unverified_context
        else:
            ssl._create_default_https_context = ssl.create_default_context

    def generate_embedding(self, chunk: str) -> list[float]:
        """
        Generates the embedding for a given chunk of text.

        Args:
            chunk (str): The input text.

        Returns:
            list[float]: The generated embedding.

        """
        self.allowSelfSignedHttps(
            True
        )  # this line is needed if you use self-signed certificate in your scoring service.

        headers, body = self.prepare_request(chunk)

        result = self.make_request(body, headers)

        return result
