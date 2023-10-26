import os
from embedding.gen_embeddings import generate_embedding
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.models import Vector  
from nlp.preprocess import Preprocess

pre_process = Preprocess()

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level


def create_client(service_endpoint, index_name, key):
    """
    Creates and returns a tuple of SearchClient and SearchIndexClient objects
    using the provided service endpoint, index name, and API key.

    Args:
        service_endpoint (str): The URL of the Azure Cognitive Search service.
        index_name (str): The name of the search index.
        key (str): The API key for the search service.

    Returns:
        Tuple[SearchClient, SearchIndexClient]: A tuple containing the SearchClient and SearchIndexClient objects.
    """
    credential = AzureKeyCredential(key)
    client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
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
        context_item['@search.score'] = result['@search.score']
        context_item['content'] =  result['content']
        formatted_results.append(context_item)

    return formatted_results


def search_for_match_semantic(client, size, query, retrieve_num_of_documents):
    """
    Searches for documents in the Azure Cognitive Search index that match the given query using semantic search.

    Args:
        client (azure.search.documents.SearchClient): The Azure Cognitive Search client.
        size (int): The size of the embedding vector.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        list: A list of formatted search results.
    """
    res = generate_embedding(size, str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle, contentSummary")  

    formatted_search_results = []
    try:
        results = client.search(  
            search_text=query,  
            vectors=[vector1, vector2], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
            query_type="semantic", query_language="en-us", semantic_configuration_name='my-semantic-config', query_caption="extractive", query_answer="extractive",
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results

# TODO: Figure out what is going on here. For some of these search functions, I cannot itterate over the results after it leaves this python file, so calling format_results on search_results which enables me to do so
# This also will provide the same format that somes back from search_for_manual_hybrid
def search_for_match_Hybrid_multi(client, size, query, retrieve_num_of_documents):
    """
    Searches for matching documents in Azure Cognitive Search using a hybrid approach that combines
    multiple vectors (contentVector, contentTitle, and contentSummary) to retrieve the most relevant
    results.

    Args:
        client (azure.search.documents.SearchClient): The Azure Cognitive Search client.
        size (int): The size of the embedding vector.
        query (str): The search query.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        list: A list of formatted search results.
    """
    res = generate_embedding(size, str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle")
    vector3 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentSummary")

    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            vectors=[vector1, vector2, vector3],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_Hybrid_cross(client, size, query, retrieve_num_of_documents):
    """
    Searches for matching documents using a hybrid cross search method.

    Args:
        client: An instance of the Azure Cognitive Search client.
        size (int): The size of the embedding.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        A list of formatted search results.
    """
    res = generate_embedding(size, str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle, contentSummary")

    formatted_search_results = []
    try:
        results = client.search(
            search_text=query,
            vectors=[vector1, vector2],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results

def search_for_match_text(client, size, query, retrieve_num_of_documents):
    """
    Searches for matching text in the given client using the specified query.

    Args:
        client: The client to search in.
        size: The size of the search.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.

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

def search_for_match_pure_vector(client, size, query, retrieve_num_of_documents):
    """
    Searches for documents in the client's database that match the given query using pure vector search.

    Args:
        client (Client): The client object used to connect to the database.
        size (int): The size of the embedding vectors.
        query (str): The query string to search for.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        A list of dictionaries containing the search results, where each dictionary represents a single document and
        contains the following keys: 'title', 'content', and 'summary'.
    """
    # function body here
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    formatted_search_results = []
    try:
        results = client.search(  
            search_text=None,  
            vectors=[vector1], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )
        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_pure_vector_multi(client, size, query, retrieve_num_of_documents):
    """
    Searches for matching documents in the given client using the provided query and retrieves the specified number of documents.

    Args:
        client: The client to search in.
        size: The size of the embedding.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.

    Returns:
        A list of formatted search results.
    """
    res = generate_embedding(size, str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle")
    vector3 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentSummary")

    formatted_search_results = []
    try:
        results = client.search(
            search_text=None,
            vectors=[vector1, vector2, vector3],
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )
        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_match_pure_vector_cross(client, size, query, retrieve_num_of_documents):
    """
    Searches for documents that match the given query using pure vector cross search method.

    Args:
        client: An instance of the search client.
        size: The size of the embedding.
        query: The query to search for.
        retrieve_num_of_documents: The number of documents to retrieve.

    Returns:
        A list of dictionaries containing the formatted search results.
    """
    # Function code here
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector, contentTitle, contentSummary")  

    formatted_search_results = []
    try:
        results = client.search(  
            search_text=None,  
            vectors=[vector1], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        logger.error(str(e))
    return formatted_search_results


def search_for_manual_hybrid(client, size, query, retrieve_num_of_documents):
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
    text_context = search_for_match_text(client, size, query, retrieve_num_of_documents)
    vector_context = search_for_match_pure_vector_cross(client, size, query, retrieve_num_of_documents)
    semantic_context = search_for_match_semantic(client, size, query, retrieve_num_of_documents)

    context.extend(text_context)
    context.extend(vector_context)
    context.extend(semantic_context)

    return context

