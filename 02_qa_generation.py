from rag_experiment_accelerator.config import Config

from dotenv import load_dotenv

load_dotenv(override=True)

from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import generate_qna

config = Config()

all_docs = load_documents(config.DATA_FORMATS, "./data/", 2000, 0)
generate_qna(all_docs, config.CHAT_MODEL_NAME, config.TEMPERATURE)
