import os
import json
from dotenv import load_dotenv

from config.config import Config

load_dotenv(override=True)

from init_Index.create_index import create_acs_index
from init_openai import init_openai
from doc_loader.pdfLoader import load_pdf_files
from embedding.gen_embeddings import generate_embedding
from ingest_data.acs_ingest import upload_data
from nlp.preprocess import Preprocess
from spacy import cli

import nltk

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)

cli.download("en_core_web_md")


def main(config: Config):
    pre_process = Preprocess()

    service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

    directory_path = 'artifacts'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    all_index_config = "artifacts/generated_index_names"
    index_dict = {"indexes": []}

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for efConstruction in config.EF_CONSTRUCTIONS:
                    for efSearch in config.EF_SEARCH:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                        print(f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}")
                        create_acs_index(service_endpoint, index_name, key, dimension, efConstruction, efSearch)
                        index_dict["indexes"].append(index_name)

    with open(all_index_config, 'w') as index_name:
        json.dump(index_dict, index_name, indent=4)

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for efConstruction in config.EF_CONSTRUCTIONS:
                    for efSearch in config.EF_SEARCH:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                        all_docs = load_pdf_files("./data/", config_item, overlap)
                        data_load = []
                        for docs in all_docs:
                            chunk_dict = {
                                "content": docs.page_content,
                                "content_vector": generate_embedding(dimension,
                                                                     str(pre_process.preprocess(docs.page_content)))
                            }
                            data_load.append(chunk_dict)
                        upload_data(data_load, service_endpoint, index_name, key, dimension,
                                    config.CHAT_MODEL_NAME, config.TEMPERATURE)


if __name__ == '__main__':
    config = Config()
    init_openai()
    main(config)
