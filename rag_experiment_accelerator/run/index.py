from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.sampling.clustering import cluster
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
        environment,
        config.CHUNKING_STRATEGY,
        config.DATA_FORMATS,
        file_paths,
        index_config.chunk_size,
        index_config.overlap,
    )

    if config.SAMPLE_DATA:
        all_docs = cluster(all_docs, config)

    data_load = []
    for doc in all_docs:
        for value in doc.values():
            chunk_dict = {
                "content": value,
                "content_vector": index_config.embedding_model.generate_embedding(
                    chunk=str(pre_process.preprocess(value))
                ),
            }
            data_load.append(chunk_dict)
    upload_data(
        environment=environment,
        config=config,
        chunks=data_load,
        index_name=index_config.index_name(),
        embedding_model=index_config.embedding_model,
    )

    return index_dict
