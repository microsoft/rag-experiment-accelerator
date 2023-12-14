import os
from rag_experiment_accelerator.config import Config

from dotenv import load_dotenv

load_dotenv(override=True)

from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import generate_qna
from rag_experiment_accelerator.run.args import get_directory_arg
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def run(config_dir: str):
    """
    Runs the main experiment loop for the QA generation process using the provided configuration and data.

    Returns:
        None
    """
    config = Config(config_dir)
    all_docs = load_documents(config.DATA_FORMATS, config.data_dir, 2000, 0)

    try:
        os.makedirs(config.artifacts_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Unable to create the '{config.artifacts_dir}' directory. Please ensure you have the proper permissions and try again")
        raise e
    
    generate_qna(all_docs, config.CHAT_MODEL_NAME, config.TEMPERATURE, config.EVAL_DATA_JSONL_FILE_PATH)
