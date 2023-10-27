from config import Config

from dotenv import load_dotenv
load_dotenv(override=True)

from doc_loader.documentLoader import load_documents
from ingest_data.acs_ingest import generate_qna

config = Config()

all_docs = load_documents(config.DATA_FORMATS, "./data/", 2000, 0)
generate_qna(all_docs, config.CHAT_MODEL_NAME, config.TEMPERATURE)
