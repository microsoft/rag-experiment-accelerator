import hashlib
import json
import traceback

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
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment

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
    environment: Environment,
    config: Config,
    chunks: list,
    index_name: str,
    embedding_model: EmbeddingModel,
):
    """
    Uploads data to an Azure AI Search index.

    Args:
        chunks (list): A list of data chunks to upload.
        service_endpoint (str): The endpoint URL for the Azure AI Search service.
        index_name (str): The name of the index to upload data to.
        search_key (str): The search key for the Azure AI Search service.
        embedding_model (EmbeddingModel): The embedding model to generate the embedding.
        azure_oai_deployment_name (str): The name of the Azure Opan AI deployment to use for generating titles and summaries.

    Returns:
        None
    """
    credential = AzureKeyCredential(environment.azure_search_admin_key)
    search_client = SearchClient(
        endpoint=environment.azure_search_service_endpoint,
        index_name=index_name,
        credential=credential,
    )
    response_generator = ResponseGenerator(
        environment, config, deployment_name=config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
    )
    documents = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_content = str(chunk["content"])
            title = response_generator.generate_response(
                prompt_instruction_title, chunk_content
            )
            summary = response_generator.generate_response(
                prompt_instruction_summary, chunk_content
            )
        except Exception as e:
            logger.info(f"Could not generate title or summary for chunk {i}: {str(e)}")
            logger.info(traceback.format_exc())
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
        if len(chunk) > 50:
            response = ""
            try:
                response_generator.generate_response(
                    generate_qna_instruction_system_prompt,
                    generate_qna_instruction_user_prompt + chunk,
                )
                response_dict = json.loads(
                    response.replace("\n", "").replace("'", "").replace("\\", "")
                )
                for item in response_dict:
                    data = {
                        "user_prompt": item["question"],
                        "output_prompt": item["answer"],
                        "context": chunk,
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


def do_we_need_multiple_questions(question, response_generator: ResponseGenerator):
    """
    Determines if we need to ask multiple questions based on the response generated by the model.

    Args:
        question (str): The question to ask.
        response_generator (ResponseGenerator): Initialised ResponseGenerator to use

    Returns:
        bool: True if we need to ask multiple questions, False otherwise.
    """
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
