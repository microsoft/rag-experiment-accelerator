from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.credentials.openai_credentials import OpenAICredentials
from openai import AzureOpenAI


class AOAIEmbeddingModel(EmbeddingModel):
    """
    A class representing an OpenAI embedding model.

    Args:
        model_name (str): The name of the model.
        openai_creds (OpenAICredentials): The OpenAI credentials.
        dimension (int, optional): The dimension of the embedding. Defaults to None.

    Attributes:
        model_name (str): The name of the model.
        dimension (int): The dimension of the embedding.
        tags (list[str]): The tags associated with the model.
        creds (OpenAICredentials): The OpenAI credentials.

    Methods:
        generate_embedding(chunk: str) -> list[float]: Generates the embedding for a given chunk of text.

    """

    def __init__(self, deployment_name: str, creds: OpenAICredentials, dimension: int = None) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(name=deployment_name, dimension=dimension)
        self._client: AzureOpenAI = self._initilize_client(creds=creds)

        
    def _initilize_client(self, creds: OpenAICredentials) -> AzureOpenAI:
        """
        Initializes the AzureOpenAIClient.

        """
        return AzureOpenAI(
                azure_endpoint=creds.OPENAI_ENDPOINT, 
                api_key=creds.OPENAI_API_KEY,  
                api_version=creds.OPENAI_API_VERSION
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
            input=chunk,
            model=self.model_name
        )

        embedding = response.data[0].embedding
        return [embedding]
