
import json
import re
import llm.prompts
from llm.prompt_execution import generate_response
import numpy as np
from sentence_transformers import CrossEncoder
from utils.logging import get_logger
logger = get_logger(__name__)

def cross_encoder_rerank_documents(documents, user_prompt, output_prompt, model_name, k):
    """
    Reranks a list of documents based on their relevance to a user prompt using a cross-encoder model.

    Args:
        documents (list): A list of documents to be reranked.
        user_prompt (str): The user prompt to be used as the query.
        output_prompt (str): The output prompt to be used as the context.
        model_name (str): The name of the pre-trained cross-encoder model to be used.
        k (int): The number of top documents to be returned.

    Returns:
        list: A list of the top k documents, sorted by their relevance to the user prompt.
    """
    model = CrossEncoder(model_name)
    cross_scores_ques = model.predict([[user_prompt, item] for item in documents],apply_softmax = True, convert_to_numpy = True )
                                    
    top_indices_ques = cross_scores_ques.argsort()[-k:][::-1]
    top_values_ques = cross_scores_ques[top_indices_ques]
    sub_context = []
    for idx in list(top_indices_ques):
        sub_context.append(documents[idx])

    return sub_context

def llm_rerank_documents(documents, question, deployment_name, temperature, rerank_threshold):
    """
    Reranks a list of documents based on a given question using the LLM model.

    Args:
        documents (list): A list of documents to be reranked.
        question (str): The question to be used for reranking.
        deployment_name (str): The name of the LLM deployment to be used.
        temperature (float): The temperature to be used for generating responses.
        rerank_threshold (int): The threshold for reranking documents.

    Returns:
        list: A list of reranked documents.
    """
    rerank_context = ""
    for index, docs in enumerate(documents):
        rerank_context += "\ndocument " + str(index) + ":\n"
        rerank_context += docs + "\n"


    prompt = f"""
        Let's try this now:
        {rerank_context}
        Question: {question}
    """

    response = generate_response(llm.prompts.rerank_prompt_instruction,prompt, deployment_name, temperature)
    logger.debug("Response", response)
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    try:
        matches = re.findall(pattern, response)[0]
        reranked = json.loads( matches )
        logger.debug(reranked)
        new_docs = []
        for key, value in reranked['documents'].items():
            key = key.replace('document_', '')
            numeric_data = re.findall(r'\d+\.\d+|\d+', key)
            if int(value) > rerank_threshold:
                new_docs.append(int(numeric_data[0]))
            result = [documents[i] for i in new_docs]
    except:
        logger.error("Unable to parse the rerank documents LLM response. Returning all documents.")
        result = documents
    return result