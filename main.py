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

# Global variables
redis_client = None  # Will be initialized in startup event
embedded_urls = set()  # Track embedded URLs

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    option: str

    class Config:
        extra = "forbid"  # Disallows extra fields

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(f"ChatRequest initialized with data: {data}")

class ChatResponse(BaseModel):
    response: str

class URLRequest(BaseModel):
    url: HttpUrl

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(f"URLRequest initialized with data: {data}")

# Middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    try:
        body = await request.json()
        logger.debug(f"Request body: {body}")
    except Exception as e:
        logger.debug(f"Could not parse request body: {e}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

# Utility Functions
def generate_random_filename(extension: str) -> str:
    filename = f"{uuid.uuid4().hex}.{extension}"
    logger.debug(f"Generated random filename: {filename}")
    return filename

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    logger.debug(f"Starting text chunking. Total length: {len(text)}")
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    logger.debug(f"Text chunking complete. Number of chunks: {len(chunks)}")
    return chunks

def extract_text_from_file(filepath: str) -> str:
    logger.debug(f"Extracting text from file: {filepath}")
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        logger.debug(f"Extracted text from PDF: {len(text)} characters.")
        return text
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        text = " ".join([p.text for p in doc.paragraphs])
        logger.debug(f"Extracted text from DOCX: {len(text)} characters.")
        return text
    else:
        logger.error("Unsupported file format. Only PDF and DOCX are supported.")
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

def save_uploaded_file(file: UploadFile) -> str:
    file_extension = file.filename.split(".")[-1]
    filename = generate_random_filename(file_extension)
    filepath = os.path.join(DOCUMENTS_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(file.file.read())
        logger.info(f"Saved uploaded file to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file.filename} to {filepath}: {e}")
        raise
    return filepath

def embed_text_chunks(text_chunks: List[str]) -> List[np.ndarray]:
    logger.debug(f"Embedding {len(text_chunks)} text chunks.")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        chunk_embeddings = embeddings.embed_documents(text_chunks)

        # Convert each embedding list to a NumPy array of type float32
        chunk_embeddings_np = [np.array(embedding).astype('float32') for embedding in chunk_embeddings]

        logger.info(f"Generated embeddings for {len(text_chunks)} chunks.")
        return chunk_embeddings_np  # List of NumPy arrays

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

def initialize_redis_vector_index():
    """Initialize Redis connection and create vector search index if it doesn't exist"""
    global redis_client
    
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=False  # Need binary mode for vector storage
        )
        
        # Check connection
        redis_client.ping()
        logger.info("Connected to Redis server successfully")
        
        # Check if index exists
        try:
            # Try to get info about the index
            redis_client.ft(REDIS_INDEX_NAME).info()
            logger.info(f"Redis index '{REDIS_INDEX_NAME}' already exists.")
        except:
            # Create vector index if it doesn't exist
            schema = (
                TextField("chunk"),
                TextField("source"),
                VectorField("embedding", 
                            "HNSW", {
                                "TYPE": "FLOAT32", 
                                "DIM": EMBEDDING_DIMENSION, 
                                "DISTANCE_METRIC": "COSINE"
                            })
            )
            
            definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
            
            redis_client.ft(REDIS_INDEX_NAME).create_index(
                fields=schema,
                definition=definition
            )
            logger.info(f"Created new Redis index '{REDIS_INDEX_NAME}'")
            
    except redis.ConnectionError as e:
        logger.critical(f"Failed to connect to Redis: {e}")
        raise RuntimeError(f"Redis connection failed: {e}")
    except Exception as e:
        logger.critical(f"Redis initialization error: {e}")
        raise RuntimeError(f"Redis initialization error: {e}")

def add_embeddings_to_redis(embeddings: List[np.ndarray], chunks: List[str], source: str = "unknown"):
    """Add embeddings and their corresponding chunks to Redis"""
    if not embeddings:
        logger.warning("No embeddings to add to Redis.")
        return
    
    # Add each chunk and its embedding to Redis
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        try:
            # Generate a unique key for this document
            doc_id = f"doc:{uuid.uuid4().hex}"
            
            # Store the document with its embedding and metadata
            redis_client.hset(
                doc_id,
                mapping={
                    "chunk": chunk,
                    "source": source,
                    "embedding": embedding.tobytes()  # Convert numpy array to bytes
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding document {i} to Redis: {e}")
            raise HTTPException(status_code=500, detail=f"Redis error: {e}")
    
    logger.info(f"Added {len(embeddings)} embeddings to Redis successfully.")