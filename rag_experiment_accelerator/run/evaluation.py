import copy
from typing import MutableMapping
from azure.ai.ml import MLClient
from dotenv import load_dotenv
import mlflow

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.evaluation import eval
from rag_experiment_accelerator.utils.logging import get_logger


load_dotenv(override=True)
logger = get_logger(__name__)


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def convert_to_dict(dictionary):
    for key, value in dictionary.items():
        dictionary[key] = value.to_dict() if hasattr(value, "to_dict") else value

    return dictionary


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def get_job_hyper_params(config: Config, index_config: IndexConfig) -> dict:
    """
    Returns the hyper parameters for the current job.
    """
    params = dict()
    config_dict = vars(copy.deepcopy(config))

    # Remove combination of hyper parameters and other not needed parameters
    for param in [
        "chunk_sizes",
        "overlap_sizes",
        "ef_constructions",
        "ef_searches",
        "embedding_models",
        "main_prompt_instruction",
        "max_worker_threads",
        "_config_dir",
        "artifacts_dir",
        "data_dir",
        "eval_data_jsonl_file_path",
        "generated_index_names_file_path",
        "query_data_location",
        "eval_data_location",
        "sampling_output_dir",
    ]:
        config_dict.__delitem__(param)

    # Add the config parameters
    params.update(flatten_dict(convert_to_dict(config_dict)))

    # Add the index config parameters by converting to dict
    params.update(flatten_dict(convert_to_dict(vars(copy.deepcopy(index_config)))))

    return params


def run(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    mlflow_client: MLClient,
    name_suffix: str,
):
    """
    Runs the evaluation process for the RAG experiment accelerator.

    This function initializes the configuration, sets up the ML client, and runs the evaluation process
    for all combinations of chunk sizes, overlap sizes, embedding dimensions, EF constructions, and EF searches.

    Returns:
        None
    """
    logger.info(f"Evaluating Index: {index_config.index_name()}")

    params = get_job_hyper_params(config, index_config)
    mlflow.log_params(params)

    eval.evaluate_prompts(
        environment=environment,
        config=config,
        index_config=index_config,
        mlflow_client=mlflow_client,
        name_suffix=name_suffix,
    )
