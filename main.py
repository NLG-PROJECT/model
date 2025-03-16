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


def retrieve_similar_chunks(prompt: str, top_k: int = 5) -> List[str]:
"""Retrieve similar text chunks from Redis using vector similarity search"""
    logger.debug(f"Retrieving similar chunks for prompt: {prompt}")
    
    try:
        # Embed the query
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        query_embedding = np.array(embeddings.embed_query(prompt)).astype('float32')
        
        # Prepare the vector search query
        vector_query = (
            f"*=>[KNN {top_k} @embedding $vector AS score]"
        )
        
        # Execute the vector search
        query = (
            Query(vector_query)
            .dialect(2)  # Use Query dialect 2 for vector search
            .sort_by("score")
            .paging(0, top_k)
            .return_fields("chunk", "score", "source")
        )
        
        params_dict = {"vector": query_embedding.tobytes()}
        
        # Execute search
        results = redis_client.ft(REDIS_INDEX_NAME).search(query, params_dict)
        logger.debug(f"Search returned {results.total} results")
        
        # Extract chunks from results
        similar_chunks = []
        for doc in results.docs:
            similar_chunks.append(doc.chunk)
            logger.debug(f"Retrieved chunk with score {doc.score}, source: {doc.source}")
        
        return similar_chunks
        
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return []

def generate_llm_response(question: str, context: str) -> str:
    """Generate a response from the LLM using Groq client with streaming."""
    try:
        # Initialize the Groq client with the API key
        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        logger.debug("Initialized Groq client.")

        prompt = (
            f"You are an AI assistant chatbot that provides detailed and comprehensive answers based on the following context.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question:\n{question}\n\n"
            f"Please provide an in-depth and thorough response around 150 words."
        )

        messages = [
            {"role": "system", "content": "You are a highly detailed-oriented and thorough assistant."},
            {"role": "user", "content": prompt},
        ]

        logger.debug("Sending completion request to Groq Cloud API.")

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.65,
            max_tokens=18890,
            top_p=0.7,
            stream=True,
            stop=None,
        )

        response_text = ""
        for chunk in completion:
            # Ensure that choices and delta exist
            if not hasattr(chunk, 'choices') or not chunk.choices:
                logger.warning("Chunk does not have 'choices'. Skipping.")
                continue

            choice = chunk.choices[0]
            if not hasattr(choice, 'delta') or not choice.delta:
                logger.warning("Choice does not have 'delta'. Skipping.")
                continue

            # Access 'content' using attribute access
            chunk_content = getattr(choice.delta, 'content', '')

            # Ensure chunk_content is a string
            if isinstance(chunk_content, str) and chunk_content:
                response_text += chunk_content
                logger.debug(f"Received chunk: {chunk_content}")
            else:
                logger.debug("No content in this chunk.")

        if not response_text.strip():
            logger.warning("LLM did not return any text in the response.")
            return "I couldn't generate a response based on the provided information."

        return response_text.strip()

    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response from Groq Cloud API.")

@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    logger.info(f"Received upload request with {len(files)} files.")
    all_chunks = []
    file_sources = []

    for idx, file in enumerate(files, 1):
        logger.debug(f"Processing file {idx}: {file.filename}")
        try:
            filepath = save_uploaded_file(file)
            content = extract_text_from_file(filepath)
            chunks = chunk_text(content)
            if not chunks:
                logger.warning(f"No text extracted from file {file.filename}. Skipping.")
                continue
            
            # Add chunks and track their source
            all_chunks.extend(chunks)
            file_sources.extend([file.filename] * len(chunks))
            logger.info(f"Extracted {len(chunks)} chunks from file {file.filename}.")
        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}: {str(e)}")

    if not all_chunks:
        logger.error("No text extracted from uploaded files.")
        raise HTTPException(status_code=400, detail="No text extracted from uploaded files.")

    try:
        new_embeddings = embed_text_chunks(all_chunks)
        
        # Add embeddings to Redis with source information
        for i, (embedding, chunk, source) in enumerate(zip(new_embeddings, all_chunks, file_sources)):
            add_embeddings_to_redis([embedding], [chunk], source)
            
    except Exception as e:
        logger.error(f"Failed during embedding or Redis operations: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed and index the uploaded files.")

    return {
        "message": "Files processed and embedded successfully.",
        "embedded_texts_count": len(all_chunks)
    }