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
from redis.commands.search.field import TextField
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from semantic_chunker import SemanticChunker  # Import our new chunker
from fastapi import Query as FastAPIQuery
import pdfplumber
from CQE.chunking.preprocessor import SECFilingPreprocessor
from CQE.chunking.structure_analyzer import StructureAnalyzer
import shutil

app = FastAPI(
    title="Document Processing API",
    description="API for processing and embedding documents",
    version="1.0.0",
    timeout=300,  # 5 minutes timeout
    max_request_size=100 * 1024 * 1024  # 100MB max request size
)
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split text into semantic chunks using the SemanticChunker.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Minimum size for each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries with text and metadata
    """
    logger.debug(f"Starting semantic text chunking. Total length: {len(text)}")
    chunker = SemanticChunker(min_chunk_size=chunk_size, overlap=overlap)
    raw_chunks = chunker.chunk(text)
    # Convert string chunks to dictionaries
    chunks = [{'text': chunk, 'context': ''} for chunk in raw_chunks]
    logger.debug(f"Semantic text chunking complete. Number of chunks: {len(chunks)}")
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

def embed_rich_chunks(chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Generate embeddings for rich chunks with text and context."""
    texts = []
    for chunk in chunks:
        text = chunk.get('text', '')
        context = chunk.get('context', '')
        combined = f"{text} {context}".strip()
        texts.append(combined)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text").embed_documents(texts)
    return [np.array(e).astype('float32') for e in embeddings]

def add_rich_chunks_to_redis(embeddings: List[np.ndarray], chunks: List[Dict[str, Any]], source: str = "unknown"):
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        chunk_id = chunk.get('id') or str(uuid.uuid4())
        chunk['id'] = chunk_id
        chunk['source'] = source
        redis_client.hset(f"chunk:{chunk_id}", mapping={"data": json.dumps(chunk)})
        redis_client.hset(f"chunk:{chunk_id}", mapping={"embedding": pickle.dumps(embedding)})

def retrieve_similar_chunks(prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
    embedding = OllamaEmbeddings(model="nomic-embed-text").embed_query(prompt)
    embedding = np.array(embedding).astype('float32')
    # Get all chunk keys
    keys = redis_client.keys("chunk:*")
    chunks = []
    for key in keys:
        chunk_data = json.loads(redis_client.hget(key, "data"))
        chunks.append(chunk_data)
    # Hybrid retrieval: prioritize chunks containing all keywords from the prompt
    keywords = [w.lower() for w in prompt.split() if len(w) > 2]
    def has_all_keywords(text):
        text_l = text.lower()
        return all(k in text_l for k in keywords)
    filtered = [c for c in chunks if has_all_keywords(c.get("text", ""))]
    if filtered:
        # If any filtered, return top_k of those
        return filtered[:top_k]
    # Otherwise, fallback to original (first top_k)
    return chunks[:top_k]

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
    all_tables = []

    for idx, file in enumerate(files, 1):
        logger.info(f"Starting to process file {idx}: {file.filename}")
        try:
            logger.info(f"Saving file {file.filename}...")
            filepath = save_uploaded_file(file)
            logger.info(f"File saved to {filepath}")

            # Extract tables using Camelot for PDFs
            tables = []
            if filepath.endswith(".pdf"):
                logger.info(f"Extracting tables from {filepath} using Camelot (stream)...")
                table_dir = "extracted_tables_json"
                # Clear the output directory before extraction
                if os.path.exists(table_dir):
                    shutil.rmtree(table_dir)
                os.makedirs(table_dir, exist_ok=True)
                # Call the table_extractor.py script as a subprocess
                subprocess.run([
                    sys.executable, "CQE/chunking/table_extractor.py", filepath, table_dir, "stream", "all"
                ], check=True)
                # Collect all JSON files created
                for fname in os.listdir(table_dir):
                    if fname.startswith("table_") and fname.endswith(".json"):
                        json_path = os.path.join(table_dir, fname)
                        with open(json_path, 'r', encoding='utf-8') as f:
                            table_data = json.load(f)
                        table_chunk = {
                            'text': json.dumps(table_data),
                            'chunk_type': 'table',
                            'source': file.filename,
                            'table_meta': {
                                'json_path': json_path,
                                'table_file': fname
                            }
                        }
                        all_chunks.append(table_chunk)
                        file_sources.append(file.filename)
                        all_tables.append(json_path)
                        logger.info(f"Added table chunk from {json_path}")

            # Extract narrative text
            if filepath.endswith(".pdf"):
                logger.info(f"Extracting narrative text from {filepath}...")
                content = extract_text_from_file(filepath)
            else:
                logger.info(f"Extracting text from {filepath}...")
                content = extract_text_from_file(filepath)

            logger.info(f"Extracted {len(content)} characters from file")

            # Preprocess narrative text
            preprocessor = SECFilingPreprocessor()
            cleaned_text = preprocessor.clean_text(content)

            # Structure analysis (optional, can be expanded)
            structure_analyzer = StructureAnalyzer()
            structure_info = structure_analyzer.analyze_structure(cleaned_text)
            # Optionally, add footnotes/cross-refs as chunks
            for footnote in structure_info.get('footnotes', []):
                all_chunks.append({
                    'text': footnote.get('content', ''),
                    'chunk_type': 'footnote',
                    'source': file.filename
                })
            # Add cross-references as metadata if needed

            # Chunk narrative text
            chunks = chunk_text(cleaned_text)
            logger.info(f"Created {len(chunks)} chunks from file {file.filename}")

            if not chunks and not all_tables:
                logger.warning(f"No text or tables extracted from file {file.filename}. Skipping.")
                continue

            # Add narrative chunks and track their source
            all_chunks.extend(chunks)
            file_sources.extend([file.filename] * len(chunks))
            logger.info(f"Extracted {len(chunks)} narrative chunks from file {file.filename}.")

        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}: {str(e)}")

    if not all_chunks:
        logger.error("No text or tables extracted from uploaded files.")
        raise HTTPException(status_code=400, detail="No text or tables extracted from uploaded files.")

    try:
        logger.info(f"Starting to embed {len(all_chunks)} chunks...")
        new_embeddings = embed_rich_chunks(all_chunks)
        logger.info(f"Successfully generated embeddings for {len(new_embeddings)} chunks")

        # Add embeddings to Redis with source information
        logger.info("Starting to add embeddings to Redis...")
        for i, (embedding, chunk, source) in enumerate(zip(new_embeddings, all_chunks, file_sources)):
            try:
                add_rich_chunks_to_redis([embedding], [chunk], source)
                logger.info(f"Successfully added embedding {i+1}/{len(new_embeddings)} to Redis")
            except Exception as e:
                logger.error(f"Failed to add embedding {i} to Redis: {str(e)}", exc_info=True)
                continue

    except Exception as e:
        logger.error(f"Failed during embedding or Redis operations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to embed and index the uploaded files.")

    return {
        "message": "Files processed and embedded successfully.",
        "embedded_texts_count": len(all_chunks),
        "extracted_tables_count": len(all_tables)
    }

@app.post("/sitemap")
async def sitemap(
    request: URLRequest, 
    depth: int = Query(1)
) -> Dict[str, Any]:
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=400, detail="Depth must be between 1 and 5")
    
    # Initialize tracking variables
    visited_urls = set()
    url_queue = deque([(str(request.url), 0)])
    processed_urls = 0
    embedded_count = 0

    # Parse the base domain to ensure we stay within the same site
    base_url = str(request.url)
    base_domain = urlparse(base_url).netloc
    base_scheme = urlparse(base_url).scheme

    while url_queue:
        current_url, current_depth = url_queue.popleft()

        # Stop if we've exceeded the specified depth
        if current_depth > depth:
            break

        # Skip if already visited
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        logger.debug(f"Processing URL: {current_url} at depth: {current_depth}")

        try:
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(current_url, timeout=10, headers=headers)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            extracted_text = soup.get_text(separator=" ")
            cleaned_text = " ".join(extracted_text.split())

            # Skip if no meaningful text
            if not cleaned_text or len(cleaned_text) < 100:
                logger.warning(f"Insufficient text extracted from URL: {current_url}")
                continue

            # Chunk and embed the text
            chunks = chunk_text(cleaned_text)
            if chunks:
                try:
                    embeddings = embed_rich_chunks(chunks)
                    add_rich_chunks_to_redis(embeddings, chunks, current_url)
                    processed_urls += 1
                    embedded_count += len(chunks)
                    logger.info(f"Successfully embedded URL: {current_url}")
                except Exception as embed_error:
                    logger.error(f"Error embedding {current_url}: {embed_error}")
                    continue        

            # Find and queue new URLs
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                
                # Normalize the URL
                try:
                    full_url = urljoin(current_url, href)
                    parsed_url = urlparse(full_url)
                except Exception as url_error:
                    logger.warning(f"Error parsing URL {href}: {url_error}")
                    continue

                # URL filtering conditions
                conditions = [
                    parsed_url.netloc == base_domain,  # Same domain
                    parsed_url.scheme in ['http', 'https'],  # Valid scheme
                    not parsed_url.fragment,  # No fragments
                    full_url not in visited_urls,  # Not visited before
                    not any(ext in full_url for ext in ['.pdf', '.jpg', '.png', '.gif']),  # Exclude media files
                ]

                # Additional path filtering to avoid going too deep into site structure
                path_segments = parsed_url.path.strip('/').split('/')
                
                # Limit path depth and avoid certain patterns
                if (all(conditions) and 
                    len(path_segments) <= 4 and  # Limit path depth
                    not any(seg in ['tag', 'category', 'archive'] for seg in path_segments)):
                    
                    # Queue the URL if it meets all conditions
                    url_queue.append((full_url, current_depth + 1))
                    logger.debug(f"Queued URL: {full_url}")

        except requests.RequestException as req_err:
            logger.error(f"Request error for {current_url}: {req_err}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {current_url}: {e}")
            continue

    return {
        "message": "Sitemap processed and embedded successfully.",
        "processed_urls_count": processed_urls,
        "total_visited_urls": len(visited_urls),
        "embedded_chunks_count": embedded_count
    }

# Normalize URLs
def normalize_url(url: str) -> str:
    """
    Normalize URLs to remove tracking parameters and standardize format
    """
    # Remove common tracking parameters
    url = re.sub(r'(\?|&)(utm_[^&]+|ref=[^&]+)', '', url)
    
    # Remove trailing slash
    url = url.rstrip('/')
    
    return url

@app.post("/clear_embedded_urls")
async def clear_embedded_urls():
    """
    Clear the list of embedded URLs.
    Useful for starting a fresh crawl or resetting the system.
    """
    global embedded_urls
    embedded_urls.clear()
    return {"message": "Embedded URLs tracking has been reset"}

@app.get("/embedded_urls")
async def get_embedded_urls():
    """
    Retrieve the list of currently embedded URLs.
    """
    return {
        "embedded_urls": list(embedded_urls),
        "total_embedded_urls": len(embedded_urls)
    }

@app.post("/upload/urls")
async def upload_urls(request: URLRequest) -> Dict[str, Any]:
    logger.info(f"Received URL upload request: {request.url}")
    try:
        # Fetch webpage content
        logger.debug(f"Fetching content from URL: {request.url}")
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()

        # Extract clean text from HTML
        logger.debug("Extracting text from HTML content.")
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()  # Remove script and style tags
        extracted_text = soup.get_text(separator=" ")

        # Remove extra spaces and newlines
        cleaned_text = " ".join(extracted_text.split())
        logger.debug(f"Cleaned text length: {len(cleaned_text)} characters.")

        # Chunk and embed the cleaned text
        chunks = chunk_text(cleaned_text)
        if not chunks:
            logger.error("No text extracted from the URL content.")
            raise ValueError("No text extracted from the URL content.")

        embeddings = embed_rich_chunks(chunks)
        add_rich_chunks_to_redis(embeddings, chunks, str(request.url))

        logger.info(f"Processed and embedded content from URL: {request.url}")

        return {
            "message": "URL content processed and embedded successfully.",
            "embedded_texts_count": len(chunks)
        }

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error while fetching URL {request.url}: {req_err}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch the URL: {req_err}")

    except Exception as e:
        logger.error(f"Error processing URL {request.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Received chat request.")
    try:
        similar_chunks = retrieve_similar_chunks(request.message, top_k=5)
        if not similar_chunks:
            logger.info("No similar chunks found for the given question.")
            return ChatResponse(response="No relevant information found. Please upload more documents or URLs.")
        # Build a rich, structured context from chunk text and metadata
        context = ""
        for i, chunk in enumerate(similar_chunks, 1):
            context += f"--- Chunk {i} ---\n"
            context += f"Section Type: {chunk.get('section_type', '')}\n"
            context += f"Item Number: {chunk.get('item_number', '')}\n"
            context += f"Content Type: {chunk.get('content_type', '')}\n"
            context += f"Chunk Type: {chunk.get('chunk_type', '')}\n"
            context += f"Parent Section: {chunk.get('parent_section', '')}\n"
            context += f"Hierarchy: {chunk.get('hierarchy', '')}\n"
            context += f"Related Sections: {chunk.get('related_sections', '')}\n"
            context += f"Source: {chunk.get('source', '')}\n"
            context += f"Source Page: {chunk.get('source_page', '')}\n"
            context += f"Text: {chunk.get('text', '')}\n"
            # Attach footnotes if present
            if chunk.get('footnotes'):
                for j, footnote in enumerate(chunk['footnotes'], 1):
                    context += f"  Footnote {j}: {footnote.get('text', '')}\n"
            # Attach cross-references if present
            if chunk.get('cross_references'):
                for j, cref in enumerate(chunk['cross_references'], 1):
                    context += f"  Cross-Reference {j}: {cref.get('text', '')}\n"
            context += "\n"
        logger.debug(f"Combined Rich Context for LLM:\n{context}")
        response_text = generate_llm_response(request.message, context)
        if not response_text:
            logger.warning("LLM did not generate a response.")
            response_text = "I couldn't generate a response based on the provided information."
        logger.info("Generated response from LLM.")
        return ChatResponse(response=response_text)
    except HTTPException as http_exc:
        logger.error(f"HTTPException in /chat endpoint: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "AI Bot Backend with Redis Vector Database and Groq Cloud is running."}

# Ollama Process Management
ollama_process = None

def start_ollama():
    global ollama_process
    if ollama_process is None:
        logger.info("Starting Ollama 'nomic-embed-text' model...")
        try:
            # Start the Ollama model as a subprocess
            ollama_process = subprocess.Popen(
                ["ollama", "run", "nomic-embed-text"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Ollama 'nomic-embed-text' model started successfully.")
            
            # Optionally, wait for a short period to ensure the model starts properly
            time.sleep(5)
        except FileNotFoundError:
            logger.critical("Ollama is not installed or not found in PATH.")
            raise RuntimeError("Ollama is not installed or not found in PATH.")
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            raise RuntimeError(f"Failed to start Ollama: {e}")

def stop_ollama():
    global ollama_process
    if ollama_process is not None:
        logger.info("Stopping Ollama 'nomic-embed-text' model...")
        try:
            # Terminate the subprocess gracefully
            if sys.platform.startswith('win'):
                ollama_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                ollama_process.terminate()
            
            # Wait for the process to terminate
            ollama_process.wait(timeout=10)
            logger.info("Ollama 'nomic-embed-text' model stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Ollama: {e}")
            ollama_process.kill()
        finally:
            ollama_process = None

# Redis management functions
def flush_redis_data():
    """Clear all data in Redis"""
    try:
        redis_client.flushall()
        logger.info("Redis data flushed successfully")
        return {"message": "All Redis data has been cleared"}
    except Exception as e:
        logger.error(f"Error flushing Redis data: {e}")
        raise HTTPException(status_code=500, detail=f"Redis error: {e}")

@app.post("/flush_redis")
async def flush_redis():
    """Endpoint to clear all Redis data"""
    return flush_redis_data()

def initialize_redis_vector_index():
    """Initialize Redis with vector search capability"""
    global redis_client
    try:
        # Initialize Redis client
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=False  # Keep binary data for embeddings
        )
        
        # Check Redis connection
        redis_client.ping()
        logger.info("Successfully connected to Redis")
        
        # Create vector index if it doesn't exist
        try:
            # Define the schema for the vector index
            schema = (
                TextField("$.data", as_name="data"),  # Store chunk data as JSON
                VectorField("$.embedding",  # Vector field for embeddings
                    "FLAT", {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIMENSION,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            
            # Create the index
            redis_client.ft(REDIS_INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["chunk:"],
                    index_type=IndexType.JSON
                )
            )
            logger.info(f"Created Redis vector index: {REDIS_INDEX_NAME}")
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                logger.info(f"Redis vector index {REDIS_INDEX_NAME} already exists")
            else:
                raise
                
    except Exception as e:
        logger.critical(f"Failed to initialize Redis: {e}")
        raise

@app.on_event("startup")
def startup_event():
    logger.info("Application startup initiated.")
    # Start Ollama before initializing Redis
    try:
        start_ollama()
    except Exception as e:
        logger.critical(f"Failed to start Ollama during startup: {e}")
        sys.exit(1)  # Exit the application if Ollama fails to start
    
    # Initialize Redis with vector search capability
    try:
        initialize_redis_vector_index()
    except Exception as e:
        logger.critical(f"Failed to initialize Redis: {e}")
        sys.exit(1)
        
    logger.info("Application startup complete. Redis vector index initialized.")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown initiated.")
    stop_ollama()
    # Close Redis connection if it exists
    global redis_client
    if redis_client:
        redis_client.close()
        logger.info("Redis connection closed.")
    logger.info("Application shutdown complete.")

@app.get("/test")
async def test_endpoint():
    logger.info("Test endpoint accessed")
    return {"status": "ok", "message": "Server is running"}

@app.get("/search_chunks")
async def search_chunks(
    query: str = FastAPIQuery(..., description="Keywords to search for in chunk text"),
    max_results: int = FastAPIQuery(10, description="Maximum number of results to return")
) -> dict:
    """
    Search stored chunks by keyword(s) for debugging and verification.
    Returns chunks containing all keywords (case-insensitive, AND logic).
    """
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    keys = redis_client.keys("chunk:*")
    results = []
    for key in keys:
        chunk_data = json.loads(redis_client.hget(key, "data"))
        text = chunk_data.get("text", "").lower()
        if all(k in text for k in keywords):
            results.append(chunk_data)
            if len(results) >= max_results:
                break
    return {
        "query": query,
        "keywords": keywords,
        "results_count": len(results),
        "results": results
    }

def extract_text_and_tables_from_pdf(filepath: str, table_dir: str = "table_extracts") -> dict:
    """
    Extracts narrative text and tables from a PDF using pdfplumber.
    Saves each table as a JSON file in table_extracts/.
    Returns:
        {
            'text': <full narrative text>,
            'tables': [
                {
                    'page': <page_number>,
                    'table_index': <index_on_page>,
                    'json_path': <path_to_json>,
                    'shape': (rows, cols)
                }, ...
            ]
        }
    """
    os.makedirs(table_dir, exist_ok=True)
    text = ""
    tables = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            page_tables = page.extract_tables()
            for t_idx, table in enumerate(page_tables, 1):
                # Save table as JSON
                json_path = os.path.join(
                    table_dir,
                    f"{os.path.basename(filepath)}_page{page_num}_table{t_idx}.json"
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(table, f, indent=2)
                tables.append({
                    'page': page_num,
                    'table_index': t_idx,
                    'json_path': json_path,
                    'shape': (len(table), len(table[0]) if table else 0)
                })
    return {'text': text, 'tables': tables}