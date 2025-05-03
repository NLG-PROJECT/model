### PATCHED UPLOAD ENDPOINT WITH TRUE PARALLEL REDIS + ASKYOURPDF FLOW, SESSION AND LOG TRACKING + CHAT FALLBACK HANDLER + REDIS TRAINING FROM FALLBACK + SIMILARITY-BASED FALLBACK CONTROL + QUALITY FILTERING ###

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
SIMILARITY_THRESHOLD = 0.35
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

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

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
        results = redis_query.docs
        good_chunks = [doc.chunk for doc in results if float(doc.score) <= SIMILARITY_THRESHOLD]
        return good_chunks
    except Exception as e:
        log_user_event("redis_search", "error", str(e))
        return []

def fallback_askyourpdf_context(prompt: str, base_doc_id: str) -> str:
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
        context = response.json()["answer"]["message"]

        fallback_vector = np.array(OllamaEmbeddings(model="nomic-embed-text").embed_query(context)).astype('float32')
        prompt_vector = np.array(OllamaEmbeddings(model="nomic-embed-text").embed_query(prompt)).astype('float32')
        similarity = cosine_similarity(prompt_vector, fallback_vector)

        if similarity >= SIMILARITY_THRESHOLD:
            fallback_chunks = chunk_text(context)
            fallback_embeddings = embed_text_chunks(fallback_chunks)
            fallback_base = f"{base_doc_id}:askfallback"
            add_embeddings_to_redis(fallback_embeddings, fallback_chunks, fallback_base, source="askyourpdf_fallback")

        return context
    except Exception as e:
        log_user_event("askyourpdf_context", "error", str(e))
        return ""

def generate_llm_response(question: str, context: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        identity = "You are DocuGroq, a world-class AI assistant that provides concise, insightful answers to document-based questions."
        instructions = (
            "Only respond based on the provided context. Use short, structured answers where possible. Aim for 100 tokens and go up to 200 only when absolutely necessary."
        )
        prompt = (
            f"{identity}\n\nContext:\n{context}\n\nUser Question:\n{question}\n\nInstructions:\n{instructions}"
        )
        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.55,
            max_tokens=200,
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
        session = json.load(open(USER_SESSION_FILE)) if os.path.exists(USER_SESSION_FILE) else {}
        chat_history = session.get("chat_history", [])
        chat_history.append({"sender": "user", "message": request.message})

        similar_chunks = retrieve_similar_chunks(request.message, top_k=5)
        context = "\n".join(similar_chunks)

        if not context:
            log_user_event("context", "fallback")
            base_doc_id = session.get("documentId", "doc:unknown")
            context = fallback_askyourpdf_context(request.message, base_doc_id)

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
