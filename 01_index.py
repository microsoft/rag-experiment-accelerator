import os
import json
from dotenv import load_dotenv

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.utils.utils import get_index_name

load_dotenv(override=True)

from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from spacy import cli

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def main(config: Config):
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    os.makedirs("artifacts", exist_ok=True)

    all_index_config = "artifacts/generated_index_names.jsonl"
    index_dict = {"indexes": []}

    for chunk_size in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        get_index_name(
                            prefix=config.NAME_PREFIX,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            embedding_model_name=embedding_model.model_name,
                            ef_construction=ef_construction,
                            ef_search=ef_search,
                        )
                        index_name = f"{config.NAME_PREFIX}-{chunk_size}-{overlap}-{embedding_model.model_name}-{ef_construction}-{ef_search}"
                        index_name = index_name.lower()
                        logger.info(f"Creating Index: {index_name}")
                        create_acs_index(
                            service_endpoint,
                            index_name,
                            key,
                            embedding_model.get_dimension(),
                            ef_construction,
                            ef_search,
                            config.LANGUAGE["analyzers"],
                        )
                        index_dict["indexes"].append(index_name)

    with open(all_index_config, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)

    for chunk_size in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            all_docs = load_documents(
                config.DATA_FORMATS, "./data/", chunk_size, overlap
            )
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{chunk_size}-{overlap}-{embedding_model.model_name}-{ef_construction}-{ef_search}"
                        index_name = index_name.lower()
                        data_load = []
                        for docs in all_docs:
                            chunk_dict = {
                                "content": docs.page_content,
                                "content_vector": embedding_model.generate_embedding(
                                    chunk=str(pre_process.preprocess(docs.page_content))
                                ),
                            }
                            data_load.append(chunk_dict)
                        upload_data(
                            chunks=data_load,
                            service_endpoint=service_endpoint,
                            index_name=index_name,
                            search_key=key,
                            chat_model_name=config.CHAT_MODEL_NAME,
                            temperature=config.TEMPERATURE,
                            embedding_model=embedding_model,
                        )


if __name__ == "__main__":
    config = Config()
    main(config)
