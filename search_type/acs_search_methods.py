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

    try:
        results = client.search(  
            search_text=query,  
            vectors=[vector1, vector2], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
            query_type="semantic", query_language="en-us", semantic_configuration_name='my-semantic-config', query_caption="extractive", query_answer="extractive",
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results

def search_for_match_Hybrid_multi(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))


    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle")  
    vector3 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentSummary") 

    try:
        results = client.search(  
            search_text=query,  
            vectors=[vector1,vector2,vector3], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results

def search_for_match_Hybrid_cross(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle, contentSummary")  

    try:
        results = client.search(  
            search_text=query,  
            vectors=[vector1, vector2], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results

def search_for_match_text(client, size, query,retrieve_num_of_documents):

    try:
        results = client.search(  
            search_text=query,  
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results

def search_for_match_pure_vector(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  

    try:
        results = client.search(  
            search_text=None,  
            vectors=[vector1], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results


def search_for_match_pure_vector_multi(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector")  
    vector2 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentTitle")  
    vector3 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentSummary") 

    try:
        results = client.search(  
            search_text=None,  
            vectors=[vector1, vector2, vector3], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results


def search_for_match_pure_vector_cross(client, size, query,retrieve_num_of_documents):
    res = generate_embedding(size,  str(pre_process.preprocess(query)))

    vector1 = Vector(value=res[0], k=retrieve_num_of_documents, fields="contentVector, contentTitle, contentSummary")  

    try:
        results = client.search(  
            search_text=None,  
            vectors=[vector1], 
            top=retrieve_num_of_documents,
            select=["title", "content", "summary"],
        )

    except Exception as e:
        results = ""
        print(str(e))
    return results

def search_for_manual_hybrid(client, size, query,retrieve_num_of_documents):
    context = []
    text_context = search_for_match_text(client, size, query,retrieve_num_of_documents)
    vector_context = search_for_match_pure_vector_cross(client, size, query,retrieve_num_of_documents)
    semantic_context = search_for_match_semantic(client, size, query,retrieve_num_of_documents)
    
    for result in text_context:  
        context_item = {}
        context_item['@search.score'] = result['@search.score']
        context_item['content'] =  result['content']
        context.append(context_item)
    for result in vector_context:  
        context_item = {}
        context_item['@search.score'] = result['@search.score']
        context_item['content'] =  result['content']
        context.append(context_item)
    for result in semantic_context:  
        context_item = {}
        context_item['@search.score'] = result['@search.score']
        context_item['content'] =  result['content']
        context.append(context_item)

    return context
