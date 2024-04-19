from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import traceback

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
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.timetook import TimeTook
from rag_experiment_accelerator.config.environment import Environment

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
    environment: Environment,
    config: Config,
    chunks: list,
    index_name: str,
):
    """
    Uploads data to an Azure AI Search index.

    This function uploads chunks of data to a specified index in Azure AI Search.
    It uses the provided service endpoint, index name, and search key to connect to the service.
    The function also converts the chunks into index documents before uploading them.
    The upload process is done in parallel using a ThreadPoolExecutor.

    Args:
        environment (Environment): The environment configuration.
        config (Config): The configuration object.
        chunks (list): A list of dictionaries, each containing a chunk of content to be uploaded.
        index_name (str): The name of the index to upload the data to.

    Returns:
        None
    """
    credential = AzureKeyCredential(environment.azure_search_admin_key)
    search_client = SearchClient(
        endpoint=environment.azure_search_service_endpoint,
        index_name=index_name,
        credential=credential,
    )

    logger.info(f"Preparing data for upload, {len(chunks)} documents to upload")
    documents = chunks_to_index_documents(chunks)

    with ExitStack() as stack:
        with TimeTook("uploading data to Azure AI Search", logger=logger):
            executor = stack.enter_context(
                ThreadPoolExecutor(config.MAX_WORKER_THREADS)
            )

            futures = {
                executor.submit(search_client.upload_documents, [document]): document
                for document in documents
            }

            for future in as_completed(futures):
                document = futures[future]
                try:
                    future.result()
                except Exception as ex:
                    logger.error(f"Failed to upload document {document}, error: {ex}")

    logger.info(
        f"Uploaded {len(documents)} documents out of {len(chunks)} documents to Azure Search Index"
    )


def generate_qna(environment, config, docs, azure_oai_deployment_name):
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
    response_generator = ResponseGenerator(
        environment, config, azure_oai_deployment_name
    )

    for doc in docs:
        # what happens with < 50 ? Currently we are skipping them
        # But we aren't explicitly saying that stating that, should we?
        chunk = list(doc.values())[0]
        if len(chunk["content"]) > 50:
            response = ""
            try:
                response = response_generator.generate_response(
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
                    f"could not generate a valid json so moving over to next "
                    f"question! Error message: {str(e)}"
                )
                logger.error(traceback.format_exc())
                logger.debug(f"LLM Response: {response}")

    return new_df


def we_need_multiple_questions(question, response_generator: ResponseGenerator):
    """
    Generates a response to a given question using a language model with multiple prompts.

    Args:
        question (str): The question to generate a response for.
        response_generator (ResponseGenerator): Initialised ResponseGenerator to use

    Returns:
        str: The generated response.
    """
    full_prompt_instruction = (
        multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    response = response_generator.generate_response(full_prompt_instruction, "")
    return response


def do_we_need_multiple_questions(
    question, response_generator: ResponseGenerator, config: Config
):
    """
    Determines if we need to ask multiple questions based on the response generated by the model.

    Args:
        question (str): The question to ask.
        response_generator (ResponseGenerator): Initialised ResponseGenerator to use

    Returns:
        bool: True if we need to ask multiple questions, False otherwise.
    """
    if not config.CHAIN_OF_THOUGHTS:
        return False

    full_prompt_instruction = (
        do_need_multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )
    try:
        response = response_generator.generate_response(full_prompt_instruction, "")

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
    Converts chunks of content into index documents for Azure AI Search.

    This function takes a list of chunks, where each chunk is a dictionary containing various pieces of content.
    It then converts each chunk into a dictionary that's suitable for use as an index document in Azure AI Search.
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
