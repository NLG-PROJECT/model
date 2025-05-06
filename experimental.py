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
import concurrent.futures
from datetime import datetime
from redis.commands.search.field import VectorField, TextField, NumericField  # [REDIS SETUP UPDATE]
import pdfplumber
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import faiss
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from utils.constants import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_INDEX_NAME, SIMILARITY_THRESHOLD, USER_SESSION_FILE, ASKYOURPDF_BASE_URL, ASKYOURPDF_API_KEY, GROQ_CLOUD_API_KEY, DOCUMENTS_DIR, FILES_LIST_PATH, USER_LOG_FILE, EMBEDDING_DIMENSION, SECONDARY_INDEX_NAME
from utils.models import ChatRequest, ChatResponse
from prompt_engineering import ASK_YOUR_SUMMARY_PROMPT, ASK_RISK_FACTORS_SUMMARY_PROMPT
from utils.utilities import update_files_list, reset_user_session, clear_user_logs, log_user_event

FAISS_CACHE = {}

# Download punkt if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Explicitly load and assign the Punkt tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

# Redis setup
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)

async def extract_text_from_file(filepath: str) -> List[tuple[str, int]]:
    """Extract text from file."""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return [(page.extract_text(), i + 1) for i, page in enumerate(reader.pages) if page.extract_text()]
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return [(p.text, 1) for p in doc.paragraphs if p.text.strip()]
    else:
        raise ValueError("Unsupported file format")

def chunk_text(pages: list[tuple[str, int]], chunk_size: int = 500, overlap: int = 50) -> List[tuple[str, int]]:
    """Chunk text into smaller pieces."""
    chunks = []
    for text, page in pages:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append((chunk, page))
    return chunks

async def embed_text_chunks(text_chunks: list[tuple[str, int]]) -> List[np.ndarray]:
    """Embed text chunks using Ollama."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    texts_only = [chunk for chunk, _ in text_chunks]
    return [np.array(e).astype('float32') for e in embeddings.embed_documents(texts_only)]

async def add_embeddings_to_redis(embeddings: list[np.ndarray], chunks: List[tuple[str, int]], base_doc_id: str, source: str):
    """Add embeddings to Redis."""
    for i, ((chunk, page), embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{base_doc_id}:chunk:{i}"
        redis_client.hset(doc_id, mapping={
            "chunk": chunk,
            "page": page,
            "source": source,
            "embedding": embedding.tobytes()
        })

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

async def retrieve_similar_chunks(prompt: str, top_k: int = 5) -> List[str]:
    """Retrieve similar chunks from Redis."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        query_embedding = np.array(embeddings.embed_query(prompt)).astype('float32')
        query = f"*=>[KNN {top_k} @embedding $vector AS score]"
        params_dict = {"vector": query_embedding.tobytes()}
        redis_query = redis_client.ft(REDIS_INDEX_NAME).search(
            redis.commands.search.query.Query(query)
            .dialect(2)
            .sort_by("score")
            .paging(0, top_k)
            .return_fields("chunk", "score"),
            params_dict
        )
        results = redis_query.docs
        good_chunks = []
        for doc in results:
            chunk = getattr(doc, "chunk", None)
            score = float(getattr(doc, "score", 1.0))
            if chunk is not None and score <= SIMILARITY_THRESHOLD:
                good_chunks.append(chunk)
        log_user_event("retrieve_similar_chunks", "completed", f"{len(good_chunks)} chunks, good_chunks: {good_chunks} ")
        return good_chunks
   
    except Exception as e:
        log_user_event("redis_search", "error", str(e))
        return []

async def fallback_askyourpdf_context(prompt: str, base_doc_id: str) -> str:
    """Fallback to AskYourPDF for context."""
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
            fallback_chunks = chunk_text([(context, -1)])
            fallback_embeddings = embed_text_chunks(fallback_chunks)
            fallback_base = f"{base_doc_id}:askfallback"
            add_embeddings_to_redis(fallback_embeddings, fallback_chunks, fallback_base, source="askyourpdf_chat_fallback")

        return context
    except Exception as e:
        log_user_event("askyourpdf_context", "error", str(e))
        return ""

async def generate_llm_response(question: str, context: str) -> str:
    """Generate LLM response."""
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

async def chat(request: ChatRequest) -> ChatResponse:
    """Chat function."""
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

async def get_market_summary():
    """Get market summary."""
    try:
        with open(USER_SESSION_FILE, "r") as f:
            session_data = json.load(f)
        if "market_summary" in session_data:
            return {"message": "Cached market summary.", "summary": session_data["market_summary"]}
        else:
            return generate_market_summary()
    except Exception as e:
        log_user_event("market_summary", "error", f"Cache check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_risk_factors():
    """Get risk factors."""
    try:
        with open(USER_SESSION_FILE, "r") as f:
            session_data = json.load(f)
        if "risk_factors" in session_data:
            return {"message": "Cached risk factors.", "summary": session_data["risk_factors"]}
        else:
            return generate_risk_factors()
    except Exception as e:
        log_user_event("risk_factors", "error", f"Cache check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_market_summary():
    """Generate market summary."""
    try:
        log_user_event("market_summary", "started")
        with open(USER_SESSION_FILE, "r") as f:
            session_data = json.load(f)
        doc_id = session_data.get("askyourpdfdocId")
        base_doc_id = session_data.get("documentId")

        if not doc_id:
            raise HTTPException(status_code=400, detail="Document ID missing in session.")

        headers = {"x-api-key": ASKYOURPDF_API_KEY, "Content-Type": "application/json"}
        ask_url = f"{ASKYOURPDF_BASE_URL}/chat/{doc_id}?stream=False"
        request_body = [{"sender": "user", "message": ASK_YOUR_SUMMARY_PROMPT.strip()}]
        ask_response = requests.post(ask_url, headers=headers, json=request_body)

        if ask_response.status_code != 200:
            log_user_event("market_summary", "ask_error", ask_response.text)
            raise HTTPException(status_code=ask_response.status_code, detail="AskYourPDF failed.")

        context = ask_response.json()["answer"]["message"]

        # Save raw response in user session
        session_data["market_summary"] = context
        with open(USER_SESSION_FILE, "w") as f:
            json.dump(session_data, f, indent=2)

        # Train embeddings with annotation
        chunks = chunk_text([(context, -1)])
        embeddings = embed_text_chunks(chunks)
        add_embeddings_to_redis(embeddings, chunks, f"{base_doc_id}:market_summary", source="askyourpdf_market_summary")

        log_user_event("market_summary", "completed")
        return {"message": "Market summary generated.", "summary": context}

    except Exception as e:
        log_user_event("market_summary", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def generate_risk_factors():
    """Generate risk factors."""
    try:
        log_user_event("risk_factors", "started")
        with open(USER_SESSION_FILE, "r") as f:
            session_data = json.load(f)
        doc_id = session_data.get("askyourpdfdocId")
        base_doc_id = session_data.get("documentId")

        if not doc_id:
            raise HTTPException(status_code=400, detail="Document ID missing in session.")

        headers = {"x-api-key": ASKYOURPDF_API_KEY, "Content-Type": "application/json"}
        ask_url = f"{ASKYOURPDF_BASE_URL}/chat/{doc_id}?stream=False"
        request_body = [{"sender": "user", "message": ASK_RISK_FACTORS_SUMMARY_PROMPT.strip()}]
        ask_response = requests.post(ask_url, headers=headers, json=request_body)

        if ask_response.status_code != 200:
            log_user_event("risk_factors", "ask_error", ask_response.text)
            raise HTTPException(status_code=ask_response.status_code, detail="AskYourPDF failed.")

        context = ask_response.json()["answer"]["message"]

        # Save in session
        session_data["risk_factors"] = context
        with open(USER_SESSION_FILE, "w") as f:
            json.dump(session_data, f, indent=2)

        # Train embeddings with annotation
        chunks = chunk_text([(context, -1)])
        embeddings = embed_text_chunks(chunks)
        add_embeddings_to_redis(embeddings, chunks, f"{base_doc_id}:risk_factors", source="askyourpdf_risk_factors")

        log_user_event("risk_factors", "completed")
        return {"message": "Risk factors summary generated.", "summary": context}

    except Exception as e:
        log_user_event("risk_factors", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
