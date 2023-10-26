from config import Config

load_dotenv(override=True)

from doc_loader.pdfLoader import load_pdf_files
from ingest_data.acs_ingest import generate_qna

config = Config()

all_docs = load_pdf_files("./data/", 2000, 0)
generate_qna(all_docs, config.CHAT_MODEL_NAME, config.TEMPERATURE)
