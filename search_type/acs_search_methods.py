from embedding.gen_embeddings import generate_embedding
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from nlp.preprocess import Preprocess

pre_process = Preprocess()

def create_client(service_endpoint,index_name, key ):
    credential = AzureKeyCredential(key)
    client = SearchClient(endpoint=service_endpoint,index_name=index_name,credential=credential)
    index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
    return (client, index_client)

def search_for_match_semantic(client, size, query,retrieve_num_of_documents):
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
        print(str(e))
    return formatted_search_results

# TODO: Figure out what is going on here. For some of these search functions, I cannot itterate over the results after it leaves this python file, so calling format_results on search_results which enables me to do so
# This also will provide the same format that somes back from search_for_manual_hybrid
def search_for_match_Hybrid_multi(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))


    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle")  
    vector3 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentSummary") 

    formatted_search_results = []
    try:
        results = client.search(  
            search_text=query,  
            vectors=[vector1,vector2,vector3], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        print(str(e))
    return formatted_search_results


def search_for_match_Hybrid_cross(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

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
        print(str(e))
    return formatted_search_results

def search_for_match_text(client, size, query,retrieve_num_of_documents):
    formatted_search_results = []
    try:
        results = client.search(  
            search_text=query,  
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

        formatted_search_results = format_results(results)

    except Exception as e:
        print(str(e))
    return formatted_search_results

def search_for_match_pure_vector(client, size, query,retrieve_num_of_documents):
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
        print(str(e))
    return formatted_search_results


def search_for_match_pure_vector_multi(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

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
        print(str(e))
    return formatted_search_results


def search_for_match_pure_vector_cross(client, size, query,retrieve_num_of_documents):
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
        print(str(e))
    return formatted_search_results


def search_for_manual_hybrid(client, size, query,retrieve_num_of_documents):
    response = []
    text_response = search_for_match_text(client, size, query,retrieve_num_of_documents)
    vector_response = search_for_match_pure_vector_cross(client, size, query,retrieve_num_of_documents)
    semantic_response = search_for_match_semantic(client, size, query,retrieve_num_of_documents)
    response.extend(text_response)
    response.extend(vector_response)
    response.extend(semantic_response)

    return response


def format_results(results):
    formatted_results = []
    for result in results:
        context_item = {}
        context_item['@search.score'] = result['@search.score']
        context_item['content'] =  result['content']
        formatted_results.append(context_item)

    return formatted_results
