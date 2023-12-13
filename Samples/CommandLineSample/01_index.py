import os
import json
from dotenv import load_dotenv

from rag_experiment_accelerator.config import Config

load_dotenv(override=True)

from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.embedding.gen_embeddings import generate_embedding
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.nlp.preprocess import Preprocess

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """
    Runs the main experiment loop, which chunks and uploads data to Azure Cognitive Search indexes based on the configuration specified in the Config class.
    
    Returns:
        None
    """
    config = Config()
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    try:
        directory = "artifacts"
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.error(f"Unable to create the '{directory}' directory. Please ensure you have the proper permissions and try again")
        raise e

    all_index_config = "artifacts/generated_index_names.jsonl"
    index_dict = {"indexes": []}

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                        logger.info(
                            f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                        )
                        create_acs_index(
                            service_endpoint,
                            index_name,
                            key,
                            dimension,
                            ef_construction,
                            ef_search,
                            config.LANGUAGE["analyzers"],
                        )
                        index_dict["indexes"].append(index_name)

    with open(all_index_config, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                        all_docs = load_documents(
                            config.DATA_FORMATS, "./data/", config_item, overlap
                        )
                        data_load = []
                        for docs in all_docs:
                            chunk_dict = {
                                "content": docs.page_content,
                                "content_vector": generate_embedding(
                                    size=dimension,
                                    chunk=str(
                                        pre_process.preprocess(docs.page_content)
                                    ),
                                    model_name=config.EMBEDDING_MODEL_NAME,
                                ),
                            }
                            data_load.append(chunk_dict)
                        upload_data(
                            chunks=data_load,
                            service_endpoint=service_endpoint,
                            index_name=index_name,
                            search_key=key,
                            dimension=dimension,
                            chat_model_name=config.CHAT_MODEL_NAME,
                            embedding_model_name=config.EMBEDDING_MODEL_NAME,
                            temperature=config.TEMPERATURE,
                        )


if __name__ == "__main__":
    main()
