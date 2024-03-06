import os
import pandas as pd
from os.path import exists

from dotenv import load_dotenv

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import generate_qna
from rag_experiment_accelerator.utils.auth import get_default_az_cred
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.sampling.clustering import dataframe_to_chunk_dict
from rag_experiment_accelerator.sampling.clustering import cluster

load_dotenv(override=True)

logger = get_logger(__name__)


def run(config_dir: str, data_dir: str = "data", filename: str = "config.json"):
    """
    Runs the main experiment loop for the QA generation process using the provided configuration and data.

    Returns:
        None
    """
    config = Config(config_dir, filename=filename)
    azure_cred = get_default_az_cred()

    all_docs = {}
    # Check if we have already sampled
    if config.SAMPLE_DATA:
        logger.info("Running QA Generation process with sampling")
        if exists(
            f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{config.SAMPLE_OPTIMUM_K}.csv"
        ):
            df = pd.read_csv(
                f"{data_dir}/sampling/sampled_cluster_predictions_cluster_number_{config.SAMPLE_OPTIMUM_K}.csv"
            )
            all_docs = dataframe_to_chunk_dict(df)
            logger.info(
                f"Loaded sampled file {data_dir}/sampling/sampled_cluster_predictions_cluster_number_{config.SAMPLE_OPTIMUM_K}.csv"
            )
        else:
            all_docs = load_documents(
                config.CHUNKING_STRATEGY,
                config.AzureDocumentIntelligenceCredentials,
                config.DATA_FORMATS,
                config.data_dir,
                2000,
                0,
            )
            all_docs = cluster(all_docs, data_dir, config)
    else:
        all_docs = load_documents(
            config.CHUNKING_STRATEGY,
            config.AzureDocumentIntelligenceCredentials,
            config.DATA_FORMATS,
            config.data_dir,
            2000,
            0,
        )

    try:
        os.makedirs(config.artifacts_dir, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Unable to create the '{config.artifacts_dir}' directory. Please"
            " ensure you have the proper permissions and try again"
        )
        raise e

    # generate qna
    df = generate_qna(all_docs, config.AZURE_OAI_CHAT_DEPLOYMENT_NAME)
    # write to jsonl
    df.to_json(config.EVAL_DATA_JSONL_FILE_PATH, orient="records", lines=True)
    # create data asset in mlstudio
    create_data_asset(
        config.EVAL_DATA_JSONL_FILE_PATH,
        "eval_data",
        azure_cred,
        config.AzureMLCredentials,
    )
