### PATCHED UPLOAD ENDPOINT WITH PARALLEL REDIS + ASKYOURPDF FLOW, SESSION AND LOG TRACKING ###

import os
import uuid
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PyPDF2 import PdfReader
from docx import Document
import requests
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import json
import logging
import sys
import redis
import asyncio
from datetime import datetime
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

app = FastAPI()

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Config
GROQ_CLOUD_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
ASKYOURPDF_API_KEY = os.getenv("ASKYOURPDF_API_KEY")
ASKYOURPDF_BASE_URL = "https://api.askyourpdf.com/v1"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_INDEX_NAME = "embeddings_idx"
EMBEDDING_DIMENSION = 768
DOCUMENTS_DIR = 'documents'
FILES_LIST_PATH = 'files_list.json'
USER_SESSION_FILE = 'user_session.json'
USER_LOG_FILE = 'user_logs.jsonl'
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Redis init
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

# Models
class ChatRequest(BaseModel):
    message: str
    option: str
    class Config:
        extra = "forbid"

class ChatResponse(BaseModel):
    response: str

# Utils

def extract_text_from_file(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def embed_text_chunks(text_chunks: List[str]) -> List[np.ndarray]:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chunk_embeddings = embeddings.embed_documents(text_chunks)
    return [np.array(embedding).astype('float32') for embedding in chunk_embeddings]

def add_embeddings_to_redis(embeddings: List[np.ndarray], chunks: List[str], base_doc_id: str, source: str):
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        doc_id = f"{base_doc_id}:chunk:{i}"
        redis_client.hset(doc_id, mapping={
            "chunk": chunk,
            "source": source,
            "embedding": embedding.tobytes()
        })

def update_files_list(new_entry: Dict[str, Any]):
    if os.path.exists(FILES_LIST_PATH):
        with open(FILES_LIST_PATH, "r") as f:
            files_list = json.load(f)
    else:
        files_list = []

    files_list.append(new_entry)

    with open(FILES_LIST_PATH, "w") as f:
        json.dump(files_list, f, indent=2)

def reset_user_session(new_entry: Dict[str, Any]):
    session_entry = {
        **new_entry,
        "chat_history": [],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(USER_SESSION_FILE, "w") as f:
        json.dump(session_entry, f, indent=2)

def log_user_event(step: str, status: str, detail: str = ""):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "status": status,
        "detail": detail
    }
    with open(USER_LOG_FILE, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

# MAIN COMBINED ROUTE
@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    logger.info(f"Uploading {len(files)} files...")
    askyourpdf_ids = []
    metadata_entries = []

    async def process_file(file: UploadFile):
        log_user_event("upload_start", "started", file.filename)

        file_bytes = await file.read()
        ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(file_bytes)

        base_doc_id = f"doc:{uuid.uuid4().hex}"

        log_user_event("extract_text", "started", file.filename)
        content = extract_text_from_file(filepath)
        chunks = chunk_text(content)
        log_user_event("extract_text", "completed", f"{len(chunks)} chunks")

        log_user_event("embedding", "started")
        embeddings = embed_text_chunks(chunks)
        add_embeddings_to_redis(embeddings, chunks, base_doc_id, file.filename)
        log_user_event("embedding", "completed")

        ask_doc_id = None
        log_user_event("askyourpdf_upload", "started")
        try:
            response = requests.post(
                f"{ASKYOURPDF_BASE_URL}/api/upload",
                headers={"x-api-key": ASKYOURPDF_API_KEY},
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            if response.status_code == 201:
                ask_doc_id = response.json().get("docId")
                log_user_event("askyourpdf_upload", "completed", ask_doc_id)
            else:
                log_user_event("askyourpdf_upload", "failed", response.text)
        except Exception as e:
            log_user_event("askyourpdf_upload", "error", str(e))

        metadata_entry = {
            "documentId": base_doc_id,
            "documentName": file.filename,
            "askyourpdfdocId": ask_doc_id
        }

        update_files_list(metadata_entry)
        reset_user_session(metadata_entry)
        log_user_event("session_reset", "completed")

        return len(chunks)

    tasks = [process_file(file) for file in files]
    chunk_counts = await asyncio.gather(*tasks)
    total_chunks = sum(chunk_counts)

    return {
        "message": "Files processed for both Redis and AskYourPDF.",
        "embedded_chunks_count": total_chunks
    }
