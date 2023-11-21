import json
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from rag_experiment_accelerator.llm.prompts import (
    prompt_instruction_title,
    prompt_instruction_summary,
    generate_qna_instruction_system_prompt,
    generate_qna_instruction_user_prompt,
    multiple_prompt_instruction,
    do_need_multiple_prompt_instruction,
)
from rag_experiment_accelerator.llm.prompt_execution import generate_response
from rag_experiment_accelerator.embedding.gen_embeddings import generate_embedding
from rag_experiment_accelerator.nlp.preprocess import Preprocess
import pandas as pd

pre_process = Preprocess()


import hashlib
from rag_experiment_accelerator.utils.logging import get_logger

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


def generate_title(chunk, model_name, temperature):
    """
    Generates a title for a given chunk of text using a language model.

    Args:
        chunk (str): The input text to generate a title for.
        model_name (str): The name of the language model to use.
        temperature (float): The temperature to use when generating the title.

    Returns:
        str: The generated title.
    """
    response = generate_response(
        prompt_instruction_title, chunk, model_name, temperature
    )
    return response


def generate_summary(chunk, model_name, temperature):
    """
    Generates a summary of the given chunk of text using the specified language model.

    Args:
        chunk (str): The text to summarize.
        model_name (str): The name of the language model to use.
        temperature (float): The "temperature" parameter to use when generating the summary.

    Returns:
        str: The generated summary.
    """
    response = generate_response(
        prompt_instruction_summary, chunk, model_name, temperature
    )
    return response


def upload_data(
    chunks: list,
    service_endpoint: str,
    index_name: str,
    search_key: str,
    dimension: int,
    chat_model_name: str,
    embedding_model_name: str,
    temperature: float,
):
    """
    Uploads data to an Azure Cognitive Search index.

    Args:
        chunks (list): A list of data chunks to upload.
        service_endpoint (str): The endpoint URL for the Azure Cognitive Search service.
        index_name (str): The name of the index to upload data to.
        search_key (str): The search key for the Azure Cognitive Search service.
        dimension (int): The dimensionality of the embeddings to generate.
        chat_model_name (str): The name of the chat model to use for generating titles and summaries.
        embedding_model_name (str): The name of the embedding model to use for generating embeddings.
        temperature (float): The temperature to use when generating titles and summaries.

    Returns:
        None
    """
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=service_endpoint, index_name=index_name, credential=credential
    )
    documents = []
    for i, chunk in enumerate(chunks):
        title = generate_title(str(chunk["content"]), chat_model_name, temperature)
        summary = generate_summary(str(chunk["content"]), chat_model_name, temperature)
        input_data = {
            "id": str(my_hash(chunk["content"])),
            "title": title,
            "summary": summary,
            "content": str(chunk["content"]),
            "filename": "test",
            "contentVector": chunk["content_vector"][0],
            "contentSummary": generate_embedding(
                size=dimension,
                chunk=str(pre_process.preprocess(summary)),
                model_name=embedding_model_name,
            )[0],
            "contentTitle": generate_embedding(
                size=dimension,
                chunk=str(pre_process.preprocess(title)),
                model_name=embedding_model_name,
            )[0],
        }

        documents.append(input_data)

        search_client.upload_documents(documents)
        logger.info(f"Uploaded {len(documents)} documents")
    logger.info("all documents have been uploaded to the search index")


def generate_qna(docs, model_name, temperature):
    """
    Generates a set of questions and answers from a list of documents using a language model.

    Args:
        docs (list): A list of documents to generate questions and answers from.
        model_name (str): The name of the language model to use.
        temperature (float): The temperature to use when generating responses.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated questions, answers, and context for each document.
    """
    column_names = ["user_prompt", "output_prompt", "context"]
    new_df = pd.DataFrame(columns=column_names)

    for i, chunk in enumerate(docs):
        if len(chunk.page_content) > 50:
            response = generate_response(
                generate_qna_instruction_system_prompt,
                generate_qna_instruction_user_prompt
                + chunk.page_content
                + "\nEND OF CONTEXT",
                model_name,
                temperature,
            )
            try:
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
                    "could not generate a valid json so moving over to next question!"
                )
                logger.debug(e)
                logger.debug(f"LLM Response: {response}")

    new_df.to_json("./artifacts/eval_data.jsonl", orient="records", lines=True)


def we_need_multiple_questions(question, model_name, temperature):
    """
    Generates a response to a given question using a language model with multiple prompts.

    Args:
        question (str): The question to generate a response for.
        model_name (str): The name of the language model to use.
        temperature (float): The temperature to use when generating the response.

    Returns:
        str: The generated response.
    """
    full_prompt_instruction = (
        multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    response1 = generate_response(full_prompt_instruction, "", model_name, temperature)
    return response1


def do_we_need_multiple_questions(question, model_name, temperature):
    """
    Determines if we need to ask multiple questions based on the response generated by the model.

    Args:
        question (str): The question to ask.
        model_name (str): The name of the model to use for generating the response.
        temperature (float): The temperature to use for generating the response.

    Returns:
        bool: True if we need to ask multiple questions, False otherwise.
    """
    full_prompt_instruction = (
        do_need_multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    response1 = generate_response(full_prompt_instruction, "", model_name, temperature)
    return re.search(r"\bHIGH\b", response1.upper())
