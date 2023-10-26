import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from init_Index.create_index import create_acs_index
from doc_loader import load_documents
from embedding.gen_embeddings import generate_embedding
from ingest_data.acs_ingest import upload_data
from nlp.preprocess import Preprocess
from spacy import cli

import nltk

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)

cli.download("en_core_web_lg")

pre_process = Preprocess()

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

with open('search_config.json', 'r') as json_file:
    data = json.load(json_file)

directory_path = 'artifacts'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

chunk_sizes = data["chunking"]["chunk_size"]
overlap_size = data["chunking"]["overlap_size"]

embedding_dimensions = data["embedding_dimension"]
efConstructions = data["efConstruction"]
efsearchs = data["efsearch"]
name_prefix = data["name_prefix"]
chat_model_name = data["chat_model_name"]
all_index_config = "artifacts/generated_index_names"
temperature = data["openai_temperature"]
index_dict = {"indexes": []}
data_formats = data.get("data_formats", "all")

for config_item in chunk_sizes:
    for overlap in overlap_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    print(f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    create_acs_index(service_endpoint, index_name, key, dimension, efConstruction, efsearch)
                    index_dict["indexes"].append(index_name)

with open(all_index_config, 'w') as index_name:
    json.dump(index_dict, index_name, indent=4)

for config_item in chunk_sizes:
    for overlap in overlap_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    all_docs = load_documents(data_formats, "./data/", config_item, overlap)
                    data_load = []
                    for docs in all_docs:
                        chunk_dict = {
                            "content": docs.page_content,
                            "content_vector": generate_embedding(dimension, str(pre_process.preprocess(docs.page_content)))
                        }
                        data_load.append(chunk_dict)
                    upload_data(data_load, service_endpoint, index_name, key, dimension, chat_model_name, temperature)
