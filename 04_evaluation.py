from dotenv import load_dotenv

load_dotenv(override=True)

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.evaluation import eval
from rag_experiment_accelerator.utils.logging import get_logger
logger = get_logger(__name__)


config = Config()

for config_item in config.CHUNK_SIZES:
    for overlap in config.OVERLAP_SIZES:
        for dimension in config.EMBEDDING_DIMENSIONS:
            for efConstruction in config.EF_CONSTRUCTIONS:
                for efSearch in config.EF_SEARCHES:
                    index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}"
                    logger.info(f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efSearch}")
                    write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                    eval.evaluate_prompts(config.NAME_PREFIX, write_path, config, config_item, overlap, dimension,
                                          efConstruction, efSearch)
