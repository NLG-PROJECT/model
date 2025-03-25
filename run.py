import uvicorn
from api.v1.app import create_app
from dotenv import load_dotenv
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the .env file
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"

# Load the main .env file
load_dotenv(env_path, override=True)

# Log the environment variables for debugging
logger.info(f"Loading .env from: {env_path}")
logger.info(f"GOOGLE_DRIVE_CREDENTIALS_PATH: {os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')}")
logger.info(f"EMBEDDING_PROVIDER: {os.getenv('EMBEDDING_PROVIDER')}")
logger.info(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")
logger.info(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")

# Ensure required environment variables are set
if not os.getenv('EMBEDDING_MODEL'):
    os.environ['EMBEDDING_MODEL'] = 'nomic-embed-text'
    logger.warning("EMBEDDING_MODEL not set, using default: nomic-embed-text")

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    ) 