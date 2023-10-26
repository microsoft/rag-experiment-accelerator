import json
from dotenv import load_dotenv

load_dotenv()

from evaluation import eval
from config.config import Config

config = Config()

name_prefix = config.NAME_PREFIX

for chunk_size in config.CHUNK_SIZES:
    for overlap in config.OVERLAP_SIZES:
        for dimension in config.EMBEDDING_DIMENSIONS:
            for efConstruction in config.EF_CONSTRUCTIONS:
                for efsearch in config.EF_SEARCH:
                    index_name = f"{name_prefix}-{chunk_size}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    print(f"{name_prefix}-{chunk_size}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                    eval.evaluate_prompts(name_prefix, write_path, config, chunk_size, overlap, dimension, efConstruction,
                                          efsearch)
