import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from evaluation import eval
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level

with open('search_config.json', 'r') as json_file:
    data = json.load(json_file)

chunk_sizes = data["chunking"]["chunk_size"]
overlap_size = data["chunking"]["overlap_size"]

embedding_dimensions = data["embedding_dimension"]
efConstructions = data["efConstruction"]
efsearchs = data["efsearch"]
name_prefix = data["name_prefix"]
search_variants = data["search_types"]
all_index_config = "artifacts/generated_index_names"
chat_model_name = data["chat_model_name"]
temperature = data["openai_temperature"]

for chunk_size in chunk_sizes:
    for overlap in overlap_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{chunk_size}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    logger.info(f"{name_prefix}-{chunk_size}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                    eval.evaluate_prompts(name_prefix, write_path, chunk_size, overlap, dimension, efConstruction,
                                          efsearch)
