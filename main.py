import os
import uuid
import pickle
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, HttpUrl
from PyPDF2 import PdfReader
from docx import Document  # Ensure using python-docx
import requests
from langchain_ollama import OllamaEmbeddings
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
import json
import logging
import sys
from groq import Groq  # New import for Groq client
import subprocess
import time
import signal
from fastapi import Query
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
app = FastAPI()
# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Bot Backend with Redis Vector Database and Groq Cloud Integration")

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
logger.debug(f"Loaded environment variables from {env_path}")

# Retrieve Groq Cloud API key from environment
GROQ_CLOUD_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
if not GROQ_CLOUD_API_KEY:
    logger.critical("GROQ_CLOUD_API_KEY is not set in the environment variables.")
    raise ValueError("GROQ_CLOUD_API_KEY is not set in the environment variables.")
else:
    logger.info("Groq Cloud API Key loaded successfully.")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_INDEX_NAME = "embeddings_idx"
EMBEDDING_DIMENSION = 768  # Example: 768 for many BERT-based models


# Directory paths
DOCUMENTS_DIR = 'documents'

# Ensure documents directory exists
try:
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    logger.debug(f"Ensured directory exists: {DOCUMENTS_DIR}")
except Exception as e:
    logger.error(f"Failed to create or access directory {DOCUMENTS_DIR}: {e}")
    raise