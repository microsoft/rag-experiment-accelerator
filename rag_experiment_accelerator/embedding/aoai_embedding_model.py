from openai import AzureOpenAI

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel


class AOAIEmbeddingModel(EmbeddingModel):
    """
    A class representing an AOAI Embedding Model.

    Args:
        deployment_name (str): The name of the deployment.
        environment (Environment): The initialised environment.
        dimension (int, optional): The dimension of the embedding. Defaults to 1536 which is the dimension of text-embedding-ada-002.
        **kwargs: Additional keyword arguments.

    Attributes:
        deployment_name (str): The name of the deployment.
        _client (AzureOpenAI): The initialized AzureOpenAI client.

    """

    def __init__(
        self,
        deployment_name: str,
        environment: Environment,
        dimension: int = 1536,
        **kwargs
    ) -> None:
        super().__init__(name=deployment_name, dimension=dimension, **kwargs)
        self.deployment_name = deployment_name
        self._client: AzureOpenAI = self._initilize_client(environment=environment)

    def _initilize_client(self, environment: Environment) -> AzureOpenAI:
        """
        Initializes the AzureOpenAIClient.

        Args:
            environment (Environment): The initialised environment.

        Returns:
            AzureOpenAI: The initialized AzureOpenAI client.

        """
        return AzureOpenAI(
            azure_endpoint=environment.openai_endpoint,
            api_key=environment.openai_api_key,
            api_version=environment.openai_api_version,
        )

    def generate_embedding(self, chunk: str) -> list[float]:
        """
        Generates the embedding for a given chunk of text.

        Args:
            chunk (str): The input text.

        Returns:
            list[float]: The generated embedding.

        """

        response = self._client.embeddings.create(
            input=chunk, model=self.deployment_name
        )

        embedding = response.data[0].embedding
        return embedding
