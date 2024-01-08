import hashlib
from contextlib import suppress
import json
import re

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.llm.prompts import (
    do_need_multiple_prompt_instruction,
    generate_qna_instruction_system_prompt,
    generate_qna_instruction_user_prompt,
    multiple_prompt_instruction,
    prompt_instruction_summary,
    prompt_instruction_title,
)
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.logging import get_logger

pre_process = Preprocess()


logger = get_logger(__name__)


def my_hash(s):
    """
    Returns the MD5 hash of the input string.

    Args:
        s (str): The string to be hashed.

    Returns:
        str: The MD5 hash of the input string.
    """
    return hashlib.md5(s.encode()).hexdigest()


def generate_title(chunk, azure_oai_deployment_name):
    """
    Generates a title for a given chunk of text using a language model.

    Args:
        chunk (str): The input text to generate a title for.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment to use.

    Returns:
        str: The generated title.
    """
    response = ResponseGenerator(
        deployment_name=azure_oai_deployment_name
    ).generate_response(prompt_instruction_title, chunk)
    return response


def generate_summary(chunk, azure_oai_deployment_name):
    """
    Generates a summary of the given chunk of text using the specified language model.

    Args:
        chunk (str): The text to summarize.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment to use.
    Returns:
        str: The generated summary.
    """
    response = ResponseGenerator(
        deployment_name=azure_oai_deployment_name
    ).generate_response(prompt_instruction_summary, chunk)
    return response


def upload_data(
    chunks: list,
    service_endpoint: str,
    index_name: str,
    search_key: str,
    embedding_model: EmbeddingModel,
    azure_oai_deployment_name: str,
):
    """
    Uploads data to an Azure Cognitive Search index.

    Args:
        chunks (list): A list of data chunks to upload.
        service_endpoint (str): The endpoint URL for the Azure Cognitive Search service.
        index_name (str): The name of the index to upload data to.
        search_key (str): The search key for the Azure Cognitive Search service.
        embedding_model (EmbeddingModel): The embedding model to generate the embedding.
        azure_oai_deployment_name (str): The name of the Azure Opan AI deployment to use for generating titles and summaries.

    Returns:
        None
    """
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=service_endpoint, index_name=index_name, credential=credential
    )
    documents = []
    for i, chunk in enumerate(chunks):
        try:
            title = generate_title(str(chunk["content"]), azure_oai_deployment_name)
            summary = generate_summary(str(chunk["content"]), azure_oai_deployment_name)
        except Exception as e:
            logger.info(f"Could not generate title or summary for chunk {i}")
            logger.info(e)
            continue
        input_data = {
            "id": str(my_hash(chunk["content"])),
            "title": title,
            "summary": summary,
            "content": str(chunk["content"]),
            "filename": "test",
            "contentVector": chunk["content_vector"],
            "contentSummary": embedding_model.generate_embedding(
                chunk=str(pre_process.preprocess(summary))
            ),
            "contentTitle": embedding_model.generate_embedding(
                chunk=str(pre_process.preprocess(title))
            ),
        }

        documents.append(input_data)

        search_client.upload_documents([input_data])
    logger.info(f"Uploaded {len(documents)} documents")
    logger.info("all documents have been uploaded to the search index")


def generate_qna(docs, azure_oai_deployment_name):
    """
    Generates a set of questions and answers from a list of documents using a language model.

    Args:
        docs (list): A list of documents to generate questions and answers from.
        azure_oai_deployment_name (str): The name of the Azure Opan AI deployment

    Returns:
        pandas.DataFrame: A DataFrame containing the generated questions, answers, and context for each document.
    """
    column_names = ["user_prompt", "output_prompt", "context"]
    new_df = pd.DataFrame(columns=column_names)

    for i, chunk in enumerate(docs):
        if len(chunk.page_content) > 50:
            response = ""
            try:
                response = ResponseGenerator(
                    deployment_name=azure_oai_deployment_name
                ).generate_response(
                    generate_qna_instruction_system_prompt,
                    generate_qna_instruction_user_prompt
                    + chunk.page_content
                    + "\nEND OF CONTEXT",
                )
                response_dict = json.loads(response)
                for item in response_dict:
                    if item["role"] == "user":
                        user_prompt = item["content"]
                    if item["role"] == "assistant":
                        output_prompt = item["content"]

                data = {
                    "user_prompt": user_prompt,
                    "output_prompt": output_prompt,
                    "context": chunk.page_content,
                }
                new_df = new_df._append(data, ignore_index=True)
                logger.info(f"Generated QnA for document {i}")
            except Exception as e:
                logger.error(
                    "could not generate a valid json so moving over to next"
                    " question!"
                )
                logger.debug(e)
                logger.debug(f"LLM Response: {response}")

    return new_df


def we_need_multiple_questions(question, azure_oai_deployment_name):
    """
    Generates a response to a given question using a language model with multiple prompts.

    Args:
        question (str): The question to generate a response for.
        azure_oai_deployment_name (str): The name of the Azure Opan AI deployment

    Returns:
        str: The generated response.
    """
    full_prompt_instruction = (
        multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    response = ResponseGenerator(
        deployment_name=azure_oai_deployment_name
    ).generate_response(full_prompt_instruction, "")
    return response


def do_we_need_multiple_questions(question, azure_oai_deployment_name):
    """
    Determines if we need to ask multiple questions based on the response generated by the model.

    Args:
        question (str): The question to ask.
        azure_oai_deployment_name (str): The name of the Azure Opan AI deployment.

    Returns:
        bool: True if we need to ask multiple questions, False otherwise.
    """
    full_prompt_instruction = (
        do_need_multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    try:
        response = ResponseGenerator(
            deployment_name=azure_oai_deployment_name
        ).generate_response(full_prompt_instruction, "")
    except ContentFilteredException as e:
        logger.error(e)
        return False
    return re.search(r"\bHIGH\b", response.upper())
