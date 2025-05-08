import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Redis setup
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_INDEX_NAME = "embeddings_idx"
EMBEDDING_DIMENSION = 768

# Constants
SIMILARITY_THRESHOLD = 0.35
USER_SESSION_FILE = 'user_session.json'
ASKYOURPDF_BASE_URL = "https://api.askyourpdf.com/v1"
ASKYOURPDF_API_KEY = os.getenv("ASKYOURPDF_API_KEY")
GROQ_CLOUD_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
DOCUMENTS_DIR = 'documents'
FILES_LIST_PATH = 'files_list.json'
USER_LOG_FILE = 'user_logs.jsonl'
SECONDARY_INDEX_NAME = "sentences_idx" 

PDF_PATH = "sample.pdf"
OCR_OUTPUT_DIR = "ocr_outputs"
TARGET_HEADINGS = {
    "consolidated statements of operations": "income_statement",
    "consolidated statements of income": "income_statement",
    "consolidated balance sheets": "balance_sheet",
    "consolidated statements of cash flows": "cash_flows",
    "consolidated statements of comprehensive income": "comprehensive_income",
    "consolidated statements of stockholders' equity": "equity_statement"
}