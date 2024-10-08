from openai import AzureOpenAI

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel


class AOAIEmbeddingModel(EmbeddingModel):
    """
    A class representing an AOAI Embedding Model.

    Args:
        model_name (str): The name of the deployment.
        environment (Environment): The initialized environment.
        dimension (int, optional): The dimension of the embedding. Defaults to 1536 which is the dimension of text-embedding-ada-002.
        **kwargs: Additional keyword arguments.

    Attributes:
        model_name (str): The name of the deployment.
        _client (AzureOpenAI): The initialized AzureOpenAI client.

    """

    def __init__(
        self,
        model_name: str,
        environment: Environment,
        dimension: int = 1536,
        shorten_dimensions: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name=model_name, dimension=dimension, **kwargs)
        self.model_name = model_name
        self.shorten_dimensions = shorten_dimensions
        self._client: AzureOpenAI = self._initialize_client(environment=environment)

    def _initialize_client(self, environment: Environment) -> AzureOpenAI:
        """
        Initializes the AzureOpenAIClient.

        Args:
            environment (Environment): The initialized environment.

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

        kwargs = {}
        if self.shorten_dimensions:
            kwargs["dimensions"] = self.dimension

        response = self._client.embeddings.create(
            input=chunk, model=self.model_name, **kwargs
        )

        return response.data[0].embedding
