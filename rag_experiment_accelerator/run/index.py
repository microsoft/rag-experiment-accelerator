from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import ntpath

from dotenv import load_dotenv

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.llm.prompts import (
    prompt_instruction_title,
    prompt_instruction_summary,
)
from rag_experiment_accelerator.sampling.clustering import cluster, load_parser
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.timetook import TimeTook
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)
load_dotenv(override=True)


def run(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    file_paths: list[str],
) -> dict[str]:
    """
    Runs the main experiment loop, which chunks and uploads data to Azure AI Search indexes based on the configuration specified in the Config class.

    Returns:
        Index dictionary containing the names of the indexes created.
    """
    pre_process = Preprocess(True)
    index_dict = {"indexes": []}

    with TimeTook(
        f"create Azure Search Index {index_config.index_name()}", logger=logger
    ):
        create_acs_index(
            environment.azure_search_service_endpoint,
            index_config.index_name(),
            environment.azure_search_admin_key,
            index_config.embedding_model.dimension,
            index_config.ef_construction,
            index_config.ef_search,
            config.LANGUAGE["analyzers"],
        )
    index_dict["indexes"].append(index_config.index_name())

    docs = load_documents(
        environment,
        config.CHUNKING_STRATEGY,
        config.DATA_FORMATS,
        file_paths,
        index_config.chunk_size,
        index_config.overlap,
        config.AZURE_DOCUMENT_INTELLIGENCE_MODEL,
    )

    if config.SAMPLE_DATA:
        parser = load_parser()
        docs = cluster(docs, config, parser)

    docs_ready_to_index = convert_docs_to_vector_db_records(docs)
    embed_chunks(index_config, pre_process, docs_ready_to_index)

    generate_titles_from_chunks(
        config, index_config, pre_process, docs_ready_to_index, environment
    )
    generate_summaries_from_chunks(
        config, index_config, pre_process, docs_ready_to_index, environment
    )

    with TimeTook(
        f"load documents to Azure Search index {index_config.index_name()}",
        logger=logger,
    ):
        upload_data(
            environment=environment,
            config=config,
            chunks=docs_ready_to_index,
            index_name=index_config.index_name(),
        )

    return index_dict


def convert_docs_to_vector_db_records(docs):
    """
    Converts a list of documents into a list of dictionaries ready to be loaded into Azure Search.

    This function takes a list of documents and converts each one into a dictionary.
    The dictionary contains the document's content and metadata.

    Args:
        docs (list): A list of documents to be converted.

    Returns:
        list: A list of dictionaries, each representing a document.
    """
    dicts = []
    for doc in docs:
        doc_id = list(doc.keys())[0]
        doc_dict = doc[doc_id]
        filename = ntpath.basename(doc_dict["metadata"].get("source", ""))
        page = doc_dict["metadata"].get("page", None)
        dict = {
            "id": doc_id,
            "content": doc_dict.get("content", ""),
            "filename": filename,
            "source_display_name": f"{filename}#page={page}"
            if str(page).isnumeric()
            else filename,
        }
        dicts.append(dict)
    return dicts


def embed_chunks(config: IndexConfig, pre_process, chunks):
    """
    Generates embeddings for chunks of documents.

    Args:
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        chunks (list): A list of all documents chunks to be embeded.

    Returns:
        tuple: A tuple containing the index name and the list of processed documents.
    """
    with TimeTook(f"generate embeddings for {config.index_name()} ", logger=logger):
        embedded_chunks = []
        with ExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor())

            futures = {
                executor.submit(
                    embed_chunk, pre_process, config.embedding_model, doc
                ): doc
                for doc in chunks
            }

            for future in as_completed(futures):
                doc = futures[future]
                try:
                    chunk_dict = future.result()
                except Exception as exc:
                    logger.error(
                        f"{embed_chunk.__name__} generated an exception: {exc} for doc {doc}"
                    )
                else:
                    embedded_chunks.append(chunk_dict)

    if config.override_content_with_summary:
        for chunk in chunks:
            if "summary" in chunk:
                chunk["content"] = chunk["summary"]
                chunk["content_vector"] = chunk["summary_vector"]
                chunk["summary"] = ""
                chunk["summary_vector"] = []
            else:
                logger.warn("summary was not generated")

    return embedded_chunks


def embed_chunk(pre_process, embedding_model, chunk):
    """
    Generates an embedding for a chunk of content.

    This function takes a chunk of content, preprocess it and generates an
    embedding for it using the `generate_embedding` function.
    The generated embedding is then added to the chunk dictionary under the
    key "content_vector".

    Args:
        pre_process (object): An object with a method for preprocessing text.
        embedding_model (object): The embedding model which was created using `EmbeddingModelFactory`.
        chunk (dict): A dictionary containing a chunk of content.

    Returns:
        dict: The chunk dictionary with the added "content_vector" key.
    """
    chunk["content_vector"] = embedding_model.generate_embedding(
        str(pre_process.preprocess(chunk["content"]))
    )

    return chunk


def generate_titles_from_chunks(
    config: Config, index_config: IndexConfig, pre_process, chunks, environment
):
    """
    Generates titles for each chunk of content in parallel using LLM and
    multithreading.

    This function uses a ThreadPoolExecutor to process each chunk in parallel.
    It submits a task to the executor for each chunk, which involves
    processing the title of the chunk.
    If an exception occurs during the processing of a chunk, it logs an error
    message with the exception and the first 20 characters of the chunk
    content.

    Args:
        config (object): A configuration object that holds various settings.
        index_config (object): An object that holds the index configuration settings.
        pre_process (object): An object with a method for preprocessing text.
        chunks (list): A list of dictionaries, each containing a chunk of content to be processed.
        environment (object): An object that holds the environment settings.
    """
    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(config.MAX_WORKER_THREADS))

        futures = {
            executor.submit(
                process_title, config, index_config, pre_process, chunk, environment
            ): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk = future.result()
            except Exception as exc:
                logger.error(
                    f"{process_title.__name__} generated an exception: {exc} for chunk {chunk['content'][0:20]}..."
                )


def generate_summaries_from_chunks(
    config: Config, index_config: IndexConfig, pre_process, chunks, environment
):
    """
    Generates summaries for each chunk of content in parallel using multithreading.

    This function uses a ThreadPoolExecutor to process each chunk in parallel.
    It submits a task to the executor for each chunk, which involves
    processing the summary of the chunk.
    If an exception occurs during the processing of a chunk, it logs an error
    message with the exception and the first 20 characters of the chunk content.

    Args:
        config (object): A configuration object that holds various settings.
        index_config (object): An object that holds the index configuration settings.
        pre_process (object): An object with a method for preprocessing text.
        chunks (list): A list of dictionaries, each containing a chunk of content to be processed.
        environment (object): An object that holds the environment settings.
    """
    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(config.MAX_WORKER_THREADS))

        futures = {
            executor.submit(
                process_summary, config, index_config, pre_process, chunk, environment
            ): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk = future.result()
            except Exception as exc:
                logger.error(
                    f"{process_summary.__name__} generated an exception: {exc} for chunk {chunk['content'][0:20]}...."
                )


def process_title(
    config: Config, index_config: IndexConfig, pre_process, chunk, environment
):
    """
    Processes the title of a chunk of content.

    If the generate_title configuration is set to True, a title is generated for the chunk of content and an embedding is created for it.
    If it's set to False, the title is set to an empty string and the title vector is set to an empty list.

    Args:
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        chunk (dict): A dictionary that contains the content to be processed.

    Returns:
        dict: The chunk dictionary with the added title and title vector.
    """
    if config.GENERATE_TITLE:
        title = generate_title(
            chunk["content"], config.AZURE_OAI_CHAT_DEPLOYMENT_NAME, environment, config
        )
        title_vector = index_config.embedding_model.generate_embedding(
            str(pre_process.preprocess(title))
        )
    else:
        title = ""
        title_vector = []

    chunk["title"] = title
    chunk["title_vector"] = title_vector

    return chunk


def process_summary(
    config: Config, index_config: IndexConfig, pre_process, chunk, environment
):
    """
    Processes the title of a chunk of content.

    If the generate_summary configuration is set to True,
    a summary is generated for the chunk of content and an embedding is
    created for it.
    If it's set to False, the summary is set to an empty string and the summary vector is set to an empty list.

    Args:
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        chunk (dict): A dictionary that contains the content to be processed.
        environment (object): An object that holds the environment settings.

    Returns:
        dict: The chunk dictionary with the added title and title vector.
    """
    if config.GENERATE_SUMMARY:
        summary = generate_summary(
            chunk["content"], config.AZURE_OAI_CHAT_DEPLOYMENT_NAME, environment, config
        )
        summaryVector = index_config.embedding_model.generate_embedding(
            str(pre_process.preprocess(summary))
        )
    else:
        summary = ""
        summaryVector = []

    chunk["summary"] = summary
    chunk["summary_vector"] = summaryVector

    return chunk


def generate_title(chunk, azure_oai_deployment_name, environment, config):
    """
    Generates a title for a given chunk of text using a language model.

    Args:
        chunk (str): The input text to generate a title for.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment to use.
        environment (object): An object that holds the environment settings.
        config (object): An object that holds the configuration settings.
    Returns:
        str: The generated title.
    """
    response = ResponseGenerator(
        environment=environment,
        config=config,
        deployment_name=azure_oai_deployment_name,
    ).generate_response(prompt_instruction_title, chunk)
    return response


def generate_summary(chunk, azure_oai_deployment_name, environment, config):
    """
    Generates a summary of the given chunk of text using the specified
    language model.

    Args:
        chunk (str): The text to summarize.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment
        to use.
        environment (object): An object that holds the environment settings.
        config (object): An object that holds the configuration settings.
    Returns:
        str: The generated summary.
    """
    response = ResponseGenerator(
        environment=environment,
        config=config,
        deployment_name=azure_oai_deployment_name,
    ).generate_response(prompt_instruction_summary, chunk)
    return response
