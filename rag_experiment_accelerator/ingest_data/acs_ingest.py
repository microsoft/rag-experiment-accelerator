from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from tqdm import tqdm


import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.llm.prompts import (
    do_need_multiple_prompt_instruction,
    generate_qna_instruction_system_prompt,
    generate_qna_instruction_user_prompt,
    multiple_prompt_instruction,
)
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.timetook import TimeTook

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


def upload_data(
    chunks: list,
    service_endpoint: str,
    index_name: str,
    search_key: str,
    config: Config,
):
    """
    Uploads data to an Azure AI Search index.

    This function uploads chunks of data to a specified index in Azure
    Cognitive Search.
    It uses the provided service endpoint, index name, and search key
    to connect to the service.
    The function also converts the chunks into index documents before
    uploading them.
    The upload process is done in parallel using a ThreadPoolExecutor.

    Args:
        chunks (list): A list of data chunks to upload.
        service_endpoint (str): The endpoint URL for the Azure AI Search
        service.
        index_name (str): The name of the index to upload data to.
        search_key (str): The search key for the Azure AI Search service.
        config (Config):

    Returns:
        None
    """
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=service_endpoint, index_name=index_name, credential=credential
    )

    logger.info(f"Preparing data for upload, {len(chunks)}" " documents to upload")
    documents = chunks_to_index_documents(chunks)
    with ExitStack() as stack:
        with TimeTook("uploading data to Azure Cognitive Search", logger=logger):
            executor = stack.enter_context(
                ThreadPoolExecutor(config.MAX_WORKER_THREADS)
            )
            progress_bar = stack.enter_context(
                tqdm(
                    total=len(documents),
                    desc="upserting documents to Azure Cognitive Search",
                    unit="doc",
                    unit_scale=True,
                )
            )

            futures = {
                executor.submit(search_client.upload_documents, document): document
                for document in documents
            }

            for future in as_completed(futures):
                document = futures[future]
                try:
                    future.result()
                except Exception as ex:
                    logger.error(f"Failed to upload document {document}, error: {ex}")
                progress_bar.update(1)

    logger.info(
        f"Uploaded {len(documents)} documents"
        f"out of {len(chunks)} documents to Azure Search Index"
    )


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

    for doc in docs:
        # what happens with < 50 ? Currently we are skipping them
        # But we aren't explicitly saying that stating that, should we?
        chunk = list(doc.values())[0]
        if len(chunk["content"]) > 50:
            response = ""
            try:
                response = ResponseGenerator(
                    deployment_name=azure_oai_deployment_name
                ).generate_response(
                    generate_qna_instruction_system_prompt,
                    generate_qna_instruction_user_prompt + chunk["content"],
                )
                response_dict = json.loads(
                    response.replace("\n", "").replace("'", "").replace("\\", "")
                )
                for item in response_dict:
                    data = {
                        "user_prompt": item["question"],
                        "output_prompt": item["answer"],
                        "context": chunk["content"],
                    }
                    new_df = new_df._append(data, ignore_index=True)

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
    Determines if we need to ask multiple questions based on the response
    generated by the model.

    Args:
        question (str): The question to ask.
        azure_oai_deployment_name (str): The name of the Azure Opan AI
        deployment.

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

        json_output = json.loads(response)
        question_complexity = json_output.get("category", "")

        if question_complexity == "" or question_complexity.lower() == "simple":
            return False
        else:
            return True
    except ContentFilteredException as e:
        logger.error(e)
        return False


def chunks_to_index_documents(chunks):
    """
    Converts chunks of content into index documents for Azure Cognitive Search.

    This function takes a list of chunks, where each chunk is a dictionary
    containing various pieces of content.
    It then converts each chunk into a dictionary that's suitable for use as
    an index document in Azure Cognitive Search.
    The resulting list of index documents is then returned.

    Args:
        chunks (list): A list of dictionaries, each containing a chunk of
        content to be converted.

    Returns:
        list: A list of dictionaries, each representing an index document.
    """
    return [
        {
            "id": str(my_hash(chunk["content"])),
            "title": chunk["title"] if "title" in chunk else "",
            "summary": chunk["summary"] if "summary" in chunk else "",
            "content": str(chunk["content"]),
            "filename": chunk["filename"],
            "sourceDisplayName": chunk["source_display_name"],
            "contentVector": chunk["content_vector"]
            if "content_vector" in chunk
            else [],
            "summaryVector": chunk["summary_vector"]
            if "summary_vector" in chunk
            else [],
            "titleVector": chunk["title_vector"] if "title_vector" in chunk else [],
        }
        for chunk in chunks
    ]
