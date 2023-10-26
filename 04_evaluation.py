from dotenv import load_dotenv

load_dotenv(override=True)

from config.config import Config
from evaluation import eval


config = Config()
all_index_config = "artifacts/generated_index_names"

for config_item in config.CHUNK_SIZES:
    for overlap in config.OVERLAP_SIZES:
        for dimension in config.EMBEDDING_DIMENSIONS:
            for efConstruction in config.EF_CONSTRUCTIONS:
                for efSearch in config.EF_SEARCH:
                    index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                    print(f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}")
                    write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                    eval.evaluate_prompts(config.NAME_PREFIX, write_path, config_item, overlap, dimension,
                                          efConstruction, efSearch)
