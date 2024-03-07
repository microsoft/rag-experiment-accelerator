from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


def run(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    file_paths: list[str],
) -> dict[str]:
    """
    Runs the main experiment loop, which chunks and uploads data to Azure AI Search indexes based on the configuration specified in the Config class.

    Returns:
        None
    """
    pre_process = Preprocess()

    index_dict = {"indexes": []}

    logger.info(f"Creating Index with name: {index_config.index_name()}")
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

    all_docs = load_documents(
        config.DATA_FORMATS, file_paths, index_config.chunk_size, index_config.overlap
    )

    data_load = []
    for docs in all_docs:
        chunk_dict = {
            "content": docs.page_content,
            "content_vector": index_config.embedding_model.generate_embedding(
                chunk=str(pre_process.preprocess(docs.page_content))
            ),
        }
        data_load.append(chunk_dict)
    upload_data(
        chunks=data_load,
        service_endpoint=environment.azure_search_service_endpoint,
        index_name=index_config.index_name(),
        search_key=environment.azure_search_admin_key,
        embedding_model=index_config.embedding_model,
        azure_oai_deployment_name=config.AZURE_OAI_CHAT_DEPLOYMENT_NAME,
    )

    return index_dict
