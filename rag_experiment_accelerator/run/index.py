from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import itertools
import json
import ntpath
import os
from tqdm import tqdm

from dotenv import load_dotenv

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.timetook import TimeTook
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.llm.prompts import (
    prompt_instruction_title,
    prompt_instruction_summary,
)

load_dotenv(override=True)


logger = get_logger(__name__)


def run(config_dir: str, data_dir: str = "data", filename: str = "config.json") -> None:
    """
    Runs the indexing process based on the provided configuration.

    This function initializes the configuration, logs the configuration
    details, and sets up the preprocessor.
    It then creates the necessary directories
    and initializes an empty index dictionary.
    The function then creates search indexes based on the configuration.
    After creating the indexes, it writes the index names to a file.
    The function then splits the documents into chunks,
    generates titles and summaries for the chunks, embeds the chunks,
    and uploads them to Azure Search.

    Args:
        config_dir (str): The directory where the configuration file
        is located.
    Raises:
        Exception: If unable to create the artifacts directory.
    """
    config = Config(config_dir, data_dir, filename)
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    ensure_artifact_dir_exits(config)

    index_dict = {"indexes": []}

    for chunk_size, overlap in chunk_size_overlap_combinations(config):
        create_search_index(
            config, service_endpoint, key, index_dict, chunk_size, overlap
        )

    dump_index_dict(config, index_dict)

    for chunk_size, overlap in chunk_size_overlap_combinations(config):
        docs_chunks = load_documents_chunks(config, chunk_size, overlap)
        chunks = convert_docs_to_dicts(docs_chunks)
        for embedding_model in config.embedding_models:
            generate_titles_from_chunks(config, pre_process, embedding_model, chunks)
            generate_summaries_from_chunks(config, pre_process, embedding_model, chunks)

            for ef_construction in config.EF_CONSTRUCTIONS:
                for ef_search in config.EF_SEARCHES:
                    index_name = generate_index_name(
                        config,
                        chunk_size,
                        overlap,
                        embedding_model,
                        ef_construction,
                        ef_search,
                    )
                    embedded_chuncks = embedd_chunks(
                        config, pre_process, embedding_model, chunks, index_name
                    )
                    upload_data(
                        chunks=embedded_chuncks,
                        service_endpoint=service_endpoint,
                        index_name=index_name,
                        search_key=key,
                        config=config,
                    )


def dump_index_dict(config, index_dict):
    index_output_file = f"{config.artifacts_dir}/generated_index_names.jsonl"
    with open(index_output_file, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)


def ensure_artifact_dir_exits(config):
    try:
        os.makedirs(config.artifacts_dir, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Unable to create the '{config.artifacts_dir}' directory. Please"
            " ensure you have the proper permissions and try again"
        )
        raise e


def chunk_size_overlap_combinations(config: Config):
    """
    Generates combinations for different chunk sizes and overlaps defined
    in the configuration

    Args:
        config (object): Configuration for rag accelerator

    Returns:
        A list of tuples for each chunk size and overlap combination
    """

    return itertools.product(config.CHUNK_SIZES, config.OVERLAP_SIZES)


def create_search_index(config, service_endpoint, key, index_dict, chunk_size, overlap):
    """
    Creates a search index in Azure Search.

    This function constructs the index name based on various configuration
    parameters and then creates the index using the `create_acs_index`
    function.
    It logs the start and end of the index creation process and the time took.
    The created index name is then added to the `index_dict` dictionary.

    Args:
        config (object): A configuration object that holds various settings.
        service_endpoint (str): The Azure Search service endpoint.
        key (str): The Azure Search service key.
        index_dict (dict): A dictionary to store the created index names.
        chunk_size (int): The size of the chunks of content.
        overlap (int): The overlap size between chunks.
        dimension (int): The size of the embedding vector.
        ef_construction (int): The size of the dynamic list used during the
        construction of the HNSW graph.
        ef_search (int): The size of the dynamic list used during the search
        phase.
    """

    for embedding_model in config.embedding_models:
        for ef_construction in config.EF_CONSTRUCTIONS:
            for ef_search in config.EF_SEARCHES:
                index_name = generate_index_name(
                    config,
                    chunk_size,
                    overlap,
                    embedding_model,
                    ef_construction,
                    ef_search,
                )
                with TimeTook(f"create Azure Search Index {index_name}", logger=logger):
                    create_acs_index(
                        service_endpoint,
                        index_name,
                        key,
                        embedding_model.dimension,
                        ef_construction,
                        ef_search,
                        config.LANGUAGE["analyzers"],
                    )
                index_dict["indexes"].append(index_name)


def generate_index_name(
    config, chunk_size, overlap, embedding_model, ef_construction, ef_search
):
    """
    Generates an index name considering config parameters

    Args:
        config (object): A configuration object that holds various settings.
        chunk_size (int): The size of the chunks of content.
        overlap (int): The overlap size between chunks.
        dimension (int): The size of the embedding vector.
        ef_construction (int): The size of the dynamic list used during the
        construction of the HNSW graph.
        ef_search (int): The size of the dynamic list used during the search
        phase.

    Returns:
        index_name (str): The generated index name.
    """

    index_name = (
        f"{config.NAME_PREFIX}-cs-{chunk_size}-o-{overlap}"
        f"-ef_c-{ef_construction}-ef_s-{ef_search}"
        f"-t-{str(config.GENERATE_TITLE)}-s-{str(config.GENERATE_SUMMARY)}"
        f"-em-{embedding_model.name}".lower()
    )
    return index_name


def load_documents_chunks(config, chunk_size, overlap):
    """
    Splits documents into chunks

    Args:
        config (object): A configuration object that holds various settings.
        chunk_size (int): The size of the chunks of content.
        overlap (int): The overlap size between chunks.

    Returns:
        list: A list of all chunks of documents.
    """
    with TimeTook(
        f"load documents with chunk_size {chunk_size} and overlap {overlap}",
        logger=logger,
    ):
        docs = load_documents(
            config.CHUNKING_STRATEGY,
            config.AzureDocumentIntelligenceCredentials,
            config.DATA_FORMATS,
            config.data_dir,
            chunk_size,
            overlap,
        )

    return docs


def convert_docs_to_dicts(docs):
    """
    Converts a list of documents into a list of dictionaries.

    This function takes a list of documents and converts each one into
    a dictionary.
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


def generate_titles_from_chunks(config, pre_process, embedding_model, chunks):
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
        pre_process (object): An object with a method for preprocessing text.
        embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        chunks (list): A list of dictionaries, each containing a chunk of
        content to be processed.
    """
    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(config.MAX_WORKER_THREADS))
        progress_bar = stack.enter_context(
            tqdm(total=len(chunks), desc="generating titles", unit="chunk")
        )

        futures = {
            executor.submit(
                proccess_title, embedding_model, config, pre_process, chunk
            ): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk = future.result()
            except Exception as exc:
                logger.error(
                    f"proccess_title generated an exception: {exc} for chunk {chunk['content'][0:20]}..."
                )
            progress_bar.update(1)


def generate_summaries_from_chunks(config, pre_process, embedding_model, chunks):
    """
    Generates summaries for each chunk of content in parallel using
    multithreading.

    This function uses a ThreadPoolExecutor to process each chunk in parallel.
    It submits a task to the executor for each chunk, which involves
    processing the summary of the chunk.
    If an exception occurs during the processing of a chunk, it logs an error
    message with the exception and the first 20 characters of the chunk
    content.

    Args:
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        chunks (list): A list of dictionaries, each containing a chunk of
        content to be processed.
    """
    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor())
        progress_bar = stack.enter_context(
            tqdm(total=len(chunks), desc="generating summaries", unit="chunk")
        )

        futures = {
            executor.submit(
                proccess_summary, embedding_model, config, pre_process, chunk
            ): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk = future.result()
            except Exception as exc:
                logger.error(
                    f"proccess_summary generated an exception: {exc}"
                    f" for chunk {chunk['content'][0:20]}...."
                )
            progress_bar.update(1)


def proccess_title(embedding_model, config, pre_process, chunk):
    """
    Processes the title of a chunk of content.

    If the GENERATE_TITLE configuration is set to True, a title is generated for the chunk of content and an embedding is created for it.
    If it's set to False, the title is set to an empty string and the title vector is set to an empty list.

    Args:
        embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        chunk (dict): A dictionary that contains the content to be processed.

    Returns:
        dict: The chunk dictionary with the added title and title vector.
    """

    if config.GENERATE_TITLE:
        title = generate_title(
            chunk["content"], config.CHAT_MODEL_NAME, config.TEMPERATURE
        )
        title_vector = embedding_model.generate_embedding(
            str(pre_process.preprocess(title))
        )[0]
    else:
        title = ""
        title_vector = []

    chunk["title"] = title
    chunk["title_vector"] = title_vector

    return chunk


def proccess_summary(embedding_model, config, pre_process, chunk):
    """
    Processes the title of a chunk of content.

    If the GENERATE_SUMMARY configuration is set to True,
    a summary is generated for the chunk of content and an embedding is
    created for it.
    If it's set to False, the summary is set to an empty string and the
    summary vector is set to an empty list.

    Args:
        embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        chunk (dict): A dictionary that contains the content to be processed.

    Returns:
        dict: The chunk dictionary with the added title and title vector.
    """
    if config.GENERATE_SUMMARY:
        summary = generate_summary(
            chunk["content"], config.CHAT_MODEL_NAME, config.TEMPERATURE
        )
        summaryVector = embedding_model.generate_embedding(
            str(pre_process.preprocess(summary))
        )[0]
    else:
        summary = ""
        summaryVector = []

    chunk["summary"] = summary
    chunk["summary_vector"] = summaryVector

    return chunk


def generate_title(chunk, azure_oai_deployment_name):
    """
    Generates a title for a given chunk of text using a language model.

    Args:
        chunk (str): The input text to generate a title for.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment
        to use.

    Returns:
        str: The generated title.
    """
    response = ResponseGenerator(
        deployment_name=azure_oai_deployment_name
    ).generate_response(prompt_instruction_title, chunk)
    return response


def generate_summary(chunk, azure_oai_deployment_name):
    """
    Generates a summary of the given chunk of text using the specified
    language model.

    Args:
        chunk (str): The text to summarize.
        azure_oai_deployment_name (str): The name of Azure Open AI deployment
        to use.
    Returns:
        str: The generated summary.
    """
    response = ResponseGenerator(
        deployment_name=azure_oai_deployment_name
    ).generate_response(prompt_instruction_summary, chunk)
    return response


def embedd_chunks(config, pre_process, embedding_model, chunks, index_name):
    """
    Generates embeddings for chunks of documents.

    Args:
        config (object): A configuration object that holds various settings.
        pre_process (object): An object with a method for preprocessing text.
        embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        chunks (list): A list of all documents chunks to be embeded.
        index_name (str): The name of the index to which the data should be
        uploaded.

    Returns:
        tuple: A tuple containing the index name and the list of processed
        documents.
    """
    with TimeTook(f"generate embeddings for {index_name} ", logger=logger):
        embedded_chunks = []
        with ExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor())
            progress_bar = stack.enter_context(
                tqdm(total=len(chunks), desc="embedding chunks", unit="chunk")
            )

            futures = {
                executor.submit(embed_chunk, pre_process, embedding_model, doc): doc
                for doc in chunks
            }

            for future in as_completed(futures):
                doc = futures[future]
                try:
                    chunk_dict = future.result()
                except Exception as exc:
                    logger.error(
                        f"embed_chunk generated an exception: {exc} for doc {doc}"
                    )
                else:
                    embedded_chunks.append(chunk_dict)
                progress_bar.update(1)

    if config.OVERRIDE_CONTENT_WITH_SUMMARY:
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
                embedding_model (object): The embedding model which was created
        using `EmbeddingModelFactory`.
        chunk (dict): A dictionary containing a chunk of content.

    Returns:
        dict: The chunk dictionary with the added "content_vector" key.
    """
    chunk["content_vector"] = embedding_model.generate_embedding(
        str(pre_process.preprocess(chunk["content"]))
    )[0]

    return chunk
