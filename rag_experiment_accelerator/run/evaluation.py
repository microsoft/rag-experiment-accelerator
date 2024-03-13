from azure.ai.ml import MLClient
from dotenv import load_dotenv

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.evaluation import eval
from rag_experiment_accelerator.utils.logging import get_logger


load_dotenv(override=True)
logger = get_logger(__name__)


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

    eval.evaluate_prompts(
        environment=environment,
        config=config,
        index_config=index_config,
        client=mlflow_client,
        name_suffix=name_suffix,
    )
