import os
import json
from dotenv import load_dotenv

from config import Config

load_dotenv(override=True)

from init_Index.create_index import create_acs_index
from doc_loader.documentLoader import load_documents
from embedding.gen_embeddings import generate_embedding
from ingest_data.acs_ingest import upload_data
from nlp.preprocess import Preprocess
from spacy import cli

import nltk

from utils.logging import get_logger
logger = get_logger(__name__)

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)


def main(config: Config):
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    directory_path = 'artifacts'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    all_index_config = "artifacts/generated_index_names"
    index_dict = {"indexes": []}

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for efConstruction in config.EF_CONSTRUCTIONS:
                    for efSearch in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                        logger.info(f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}")
                        create_acs_index(service_endpoint, index_name, key, dimension, efConstruction, efSearch)
                        index_dict["indexes"].append(index_name)

    with open(all_index_config, 'w') as index_name:
        json.dump(index_dict, index_name, indent=4)

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for efConstruction in config.EF_CONSTRUCTIONS:
                    for efSearch in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                        all_docs = load_documents(
                            config.DATA_FORMATS,
                            "./data/",
                            config_item,
                            overlap,
                            config.LANGUAGE["analyzers"],
                        )
                        data_load = []
                        for docs in all_docs:
                            chunk_dict = {
                                "content": docs.page_content,
                                "content_vector": generate_embedding(
                                    size=dimension,
                                    chunk=str(pre_process.preprocess(docs.page_content)),
                                    model_name=config.EMBEDDING_MODEL_NAME
                                )
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
                            temperature=config.TEMPERATURE
                        )


if __name__ == '__main__':
    config = Config()
    main(config)
