import pandas as pd
from os.path import exists

from dotenv import load_dotenv

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import generate_qna
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.sampling.clustering import (
    dataframe_to_chunk_dict,
    load_parser,
)
from rag_experiment_accelerator.sampling.clustering import cluster

load_dotenv(override=True)

logger = get_logger(__name__)


def run(
    environment: Environment,
    config: Config,
    file_paths: list[str],
):
    """
    Runs the main experiment loop for the QA generation process using the provided configuration and data.

    Returns:
        None
    """
    logger.info("Running QA generation")

    all_docs = {}
    # Check if we have already sampled
    if config.SAMPLE_DATA:
        logger.info("Running QA Generation process with sampling")
        if exists(config._sampled_cluster_predictions_path()):
            df = pd.read_csv(config._sampled_cluster_predictions_path())
            all_docs = dataframe_to_chunk_dict(df)
            logger.info("Loaded sampled data")
        else:
            all_docs = load_documents(
                environment,
                config.CHUNKING_STRATEGY,
                config.DATA_FORMATS,
                file_paths,
                2000,
                0,
            )
            parser = load_parser()
            all_docs = cluster(all_docs, config.sampling_output_dir, config, parser)
    else:
        all_docs = load_documents(
            environment,
            config.CHUNKING_STRATEGY,
            config.DATA_FORMATS,
            file_paths,
            2000,
            0,
        )

    # generate qna
    df = generate_qna(
        environment, config, all_docs, config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
    )
    # write to jsonl
    df.to_json(config.EVAL_DATA_JSONL_FILE_PATH, orient="records", lines=True)
    # create data asset in mlstudio
    create_data_asset(config.EVAL_DATA_JSONL_FILE_PATH, "eval_data", environment)
