### PATCHED UPLOAD ENDPOINT WITH TRUE PARALLEL REDIS + ASKYOURPDF FLOW, SESSION AND LOG TRACKING + CHAT FALLBACK HANDLER ###

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

def clear_user_logs():
    with open(USER_LOG_FILE, "w") as log_file:
        log_file.write("")

def log_user_event(step: str, status: str, detail: str = ""):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "status": status,
        "detail": detail
    }
    with open(USER_LOG_FILE, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

def retrieve_similar_chunks(prompt: str, top_k: int = 5) -> List[str]:
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        query_embedding = np.array(embeddings.embed_query(prompt)).astype('float32')
        query = f"*=>[KNN {top_k} @embedding $vector AS score]"
        params_dict = {"vector": query_embedding.tobytes()}
        redis_query = redis_client.ft(REDIS_INDEX_NAME).search(
            redis.commands.search.query.Query(query).dialect(2).sort_by("score").paging(0, top_k).return_fields("chunk", "score"),
            params_dict
        )
        return [doc.chunk for doc in redis_query.docs]
    except Exception as e:
        log_user_event("redis_search", "error", str(e))
        return []

def fallback_askyourpdf_context(prompt: str) -> str:
    try:
        with open(USER_SESSION_FILE, "r") as f:
            session = json.load(f)
        doc_id = session.get("askyourpdfdocId")
        chat_history = session.get("chat_history", []) + [{"sender": "user", "message": prompt}]
        response = requests.post(
            f"{ASKYOURPDF_BASE_URL}/chat/{doc_id}?stream=False",
            headers={"x-api-key": ASKYOURPDF_API_KEY, "Content-Type": "application/json"},
            json=chat_history
        )
        return response.json()["answer"]["message"]
    except Exception as e:
        log_user_event("askyourpdf_context", "error", str(e))
        return ""

def generate_llm_response(question: str, context: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        prompt = (
            f"You are an AI assistant chatbot that provides detailed and comprehensive answers based on the following context.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question:\n{question}\n\n"
            f"Please provide an in-depth and thorough response."
        )
        messages = [
            {"role": "system", "content": "You are a highly detailed-oriented and thorough assistant."},
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.65,
            max_tokens=2048,
            top_p=0.7,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log_user_event("groq_response", "error", str(e))
        return ""

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        log_user_event("chat", "started", request.message)

        # Load session and history
        session = json.load(open(USER_SESSION_FILE)) if os.path.exists(USER_SESSION_FILE) else {}
        chat_history = session.get("chat_history", [])
        chat_history.append({"sender": "user", "message": request.message})

        similar_chunks = retrieve_similar_chunks(request.message, top_k=5)
        context = "\n".join(similar_chunks)

        if not context:
            log_user_event("context", "fallback")
            context = fallback_askyourpdf_context(request.message)

        answer = generate_llm_response(request.message, context)

        chat_history.append({"sender": "bot", "message": answer})
        session["chat_history"] = chat_history
        with open(USER_SESSION_FILE, "w") as f:
            json.dump(session, f, indent=2)

        log_user_event("chat", "completed")
        return ChatResponse(response=answer)

    except Exception as e:
        log_user_event("chat", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# MAIN COMBINED ROUTE
@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    logger.info(f"Uploading {len(files)} files...")
    clear_user_logs()
    log_user_event("upload_batch", "start", f"{len(files)} files")

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

        async def embed_and_store():
            log_user_event("embedding", "started")
            embeddings = embed_text_chunks(chunks)
            add_embeddings_to_redis(embeddings, chunks, base_doc_id, file.filename)
            log_user_event("embedding", "completed")

        async def upload_to_askyourpdf():
            ask_doc_id = None
            log_user_event("askyourpdf_upload", "started")
            try:
                response = requests.post(
                    f"{ASKYOURPDF_BASE_URL}/upload",
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
            return ask_doc_id

        ask_task = asyncio.create_task(upload_to_askyourpdf())
        redis_task = asyncio.create_task(embed_and_store())
        ask_doc_id = await ask_task
        await redis_task

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

    log_user_event("upload_batch", "complete", f"{total_chunks} chunks embedded")
    return {
        "message": "Files processed for both Redis and AskYourPDF.",
        "embedded_chunks_count": total_chunks
    }
