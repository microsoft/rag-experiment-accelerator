from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.config.credentials import OpenAICredentials
from openai import AzureOpenAI


class AOAIEmbeddingModel(EmbeddingModel):
    """
    A class representing an AOAI Embedding Model.

    Args:
        deployment_name (str): The name of the deployment.
        openai_creds (OpenAICredentials): The OpenAI credentials.
        dimension (int, optional): The dimension of the embedding. Defaults to 1536 which is the dimension of text-embedding-ada-002.
        **kwargs: Additional keyword arguments.

    Attributes:
        deployment_name (str): The name of the deployment.
        _client (AzureOpenAI): The initialized AzureOpenAI client.

    """

    def __init__(
        self,
        deployment_name: str,
        openai_creds: OpenAICredentials,
        dimension: int = 1536,
        **kwargs
    ) -> None:
        super().__init__(name=deployment_name, dimension=dimension, **kwargs)
        self.deployment_name = deployment_name
        self._client: AzureOpenAI = self._initilize_client(creds=openai_creds)

    def _initilize_client(self, creds: OpenAICredentials) -> AzureOpenAI:
        """
        Initializes the AzureOpenAIClient.

        Args:
            creds (OpenAICredentials): The OpenAI credentials.

        Returns:
            AzureOpenAI: The initialized AzureOpenAI client.

        """
        return AzureOpenAI(
            azure_endpoint=creds.OPENAI_ENDPOINT,
            api_key=creds.OPENAI_API_KEY,
            api_version=creds.OPENAI_API_VERSION,
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
