from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryLanguage,
    QueryType,
    RawVectorQuery,
)

from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

pre_process = Preprocess()


def create_client(service_endpoint, index_name, key):
    """
    Creates and returns a tuple of SearchClient and SearchIndexClient objects
    using the provided service endpoint, index name, and API key.

    Args:
        service_endpoint (str): The URL of the Azure AI Search service.
        index_name (str): The name of the search index.
        key (str): The API key for the search service.

    Returns:
        Tuple[SearchClient, SearchIndexClient]: A tuple containing the SearchClient and SearchIndexClient objects.
    """
    credential = AzureKeyCredential(key)
    client = SearchClient(
        endpoint=service_endpoint, index_name=index_name, credential=credential
    )
    return client


def format_results(results):
    """
    Formats the search results by extracting the score and content fields from each result.

    Args:
        results (list): A list of search results.

    Returns:
        list: A list of dictionaries, where each dictionary contains the score and content fields of a search result.
    """
    formatted_results = []
    for result in results:
        context_item = {}
        context_item["@search.score"] = result["@search.score"]
        context_item["content"] = result["content"]
        formatted_results.append(context_item)

    return formatted_results


def search_for_match_semantic(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for documents in the Azure AI Search index that match the given query using semantic search.

    Args:
        client (SearchClient): The Azure AI Search client.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        list: A list of formatted search results.
    """
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector",
        vector=embedding,
    )
    vector2 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="titleVector, summaryVector",
        vector=embedding,
    )

    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            vector_queries=[vector1, vector2],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
            query_type=QueryType.SEMANTIC,
            query_language=QueryLanguage.EN_US,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


# TODO: Figure out what is going on here. For some of these search functions,
# I cannot iterate over the results after it leaves this python file, so
# calling format_results on search_results which enables me to do so
# This also will provide the same format that comes back from
# search_for_manual_hybrid
def search_for_match_Hybrid_multi(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for matching documents in Azure AI Search using a hybrid approach that combines
    multiple vectors (contentVector, title, and summary) to retrieve the most relevant
    results.

    Args:
        client (SearchClient): The Azure AI Search client.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query (str): The search query.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        list: A list of formatted search results.
    """
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector",
        vector=embedding,
    )
    vector2 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="titleVector",
        vector=embedding,
    )
    vector3 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="summaryVector",
        vector=embedding,
    )

    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            vector_queries=[vector1, vector2, vector3],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_Hybrid_cross(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for matching documents using a hybrid cross search method.

    Args:
        client: An instance of the Azure AI Search client.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        A list of formatted search results.
    """
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector",
        vector=embedding,
    )
    vector2 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="titleVector, summaryVector",
        vector=embedding,
    )

    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            vector_queries=[vector1, vector2],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_text(
    client: SearchClient,
    query: str,
    retrieve_num_of_documents: int,
    **kwargs,
):
    """
    Searches for matching text in the given client using the specified query.

    Args:
        client: The client to search in.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of formatted search results.
    """
    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_pure_vector(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for documents in the client's database that match the given query using pure vector search.

    Args:
        client (Client): The client object used to connect to the database.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        A list of dictionaries containing the search results, where each dictionary represents a single document and
        contains the following keys: 'title', 'content', and 'summary'.
    """
    # function body here
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector",
        vector=embedding,
    )
    formatted_search_results = []
    try:
        results = client.search(
            search_text=None,
            vector_queries=[vector1],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )
        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_pure_vector_multi(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for matching documents in the given client using the provided query and retrieves the specified number
    of documents.

    Args:
        client: The client to search in.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.

    Returns:
        A list of formatted search results.
    """
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector",
        vector=embedding,
    )
    vector2 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="title",
        vector=embedding,
    )
    vector3 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="summary",
        vector=embedding,
    )

    formatted_search_results = []
    try:
        results = client.search(
            search_text=None,
            vector_queries=[vector1, vector2, vector3],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )
        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_pure_vector_cross(
    client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    retrieve_num_of_documents: int,
):
    """
    Searches for documents that match the given query using pure vector cross search method.

    Args:
        client: An instance of the search client.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.

    Returns:
        A list of dictionaries containing the formatted search results.
    """
    # Function code here
    embedding = embedding_model.generate_embedding(
        chunk=str(pre_process.preprocess(query))
    )

    vector1 = RawVectorQuery(
        k=retrieve_num_of_documents,
        fields="contentVector, titleVector, summaryVector",
        vector=embedding,
    )

    formatted_search_results = []
    try:
        results = client.search(
            search_text=None,
            vector_queries=[vector1],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_manual_hybrid(**kwargs):
    """
    Searches for documents using a combination of text, vector, and semantic matching.

    Args:
        client: Elasticsearch client object.
        size: Maximum number of documents to retrieve.
        query: Query string to search for.
        retrieve_num_of_documents: Number of documents to retrieve.

    Returns:
        A list of documents matching the search query.
    """
    context = []
    text_context = search_for_match_text(**kwargs)
    vector_context = search_for_match_pure_vector_cross(**kwargs)
    semantic_context = search_for_match_semantic(**kwargs)

    context.extend(text_context)
    context.extend(vector_context)
    context.extend(semantic_context)

    return context
