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
from experimental import extract_text_from_file, chunk_text, embed_text_chunks, add_embeddings_to_redis, log_user_event
from utils.constants import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_INDEX_NAME, SIMILARITY_THRESHOLD, USER_SESSION_FILE, ASKYOURPDF_BASE_URL, ASKYOURPDF_API_KEY, GROQ_CLOUD_API_KEY, DOCUMENTS_DIR, FILES_LIST_PATH, USER_LOG_FILE, EMBEDDING_DIMENSION, SECONDARY_INDEX_NAME
from utils.models import ChatRequest, ChatResponse
from prompt_engineering import ASK_YOUR_SUMMARY_PROMPT, ASK_RISK_FACTORS_SUMMARY_PROMPT
from utils.utilities import update_files_list, reset_user_session, clear_user_logs
from utils.constants import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_INDEX_NAME, SIMILARITY_THRESHOLD, USER_SESSION_FILE, ASKYOURPDF_BASE_URL, ASKYOURPDF_API_KEY, GROQ_CLOUD_API_KEY, DOCUMENTS_DIR
from utils.models import FactCheckRequest

FAISS_CACHE = {}
# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

# Redis setup
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)

def base_fact_check_answer(statement: str, threshold: float = 0.7) -> Dict[str, Any]:
    """Base fact check answer function."""
    from nltk.tokenize import PunktSentenceTokenizer
    if not os.path.exists(USER_SESSION_FILE):
        return {"error": "No session."}
    
    session = json.load(open(USER_SESSION_FILE))
    doc_id = session.get("documentId")
    if not doc_id:
        return {"error": "No document ID."}

    raw_file_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}_raw.txt")
    if not os.path.exists(raw_file_path):
        return {"error": "No raw document."}

    with open(raw_file_path, encoding="utf-8") as f:
        raw_text = f.read()

    # Split into [(page, sentence)]
    pages = raw_text.split("[PAGE ")
    sentence_page_map = []
    tokenizer = PunktSentenceTokenizer()

    for page_text in pages[1:]:
        try:
            page_num, content = page_text.split("]", 1)
            page_num = int(page_num.strip())
            sentences = tokenizer.tokenize(content)
            for sent in sentences:
                sentence_page_map.append((sent.strip(), page_num))
        except Exception:
            continue

    claims = tokenizer.tokenize(statement)
    embed = OllamaEmbeddings(model="nomic-embed-text")

    doc_sentences = [s for s, _ in sentence_page_map]
    doc_vectors = embed.embed_documents(doc_sentences)
    claim_vectors = embed.embed_documents(claims)

    results = []
    for i, vec in enumerate(claim_vectors):
        best_idx, best_score = -1, 0.0
        for j, doc_vec in enumerate(doc_vectors):
            score = float(np.dot(vec, doc_vec) / (np.linalg.norm(vec) * np.linalg.norm(doc_vec)))
            if score > best_score:
                best_score = score
                best_idx = j
        sentence, page = sentence_page_map[best_idx] if best_idx != -1 else ("", None)
        results.append({
            "claim": claims[i],
            "score": best_score,
            "status": "supported" if best_score >= threshold else "unsupported",
            "evidence": sentence if best_score >= threshold else None,
            "page": page if best_score >= threshold else None
        })

    return {"fact_check": results}

def fact_check(request: FactCheckRequest):
    """Fact check function."""
    import logging
    logger = logging.getLogger("FactCheck")
    logger.info("Starting FAISS fact check...")

    if not os.path.exists(USER_SESSION_FILE):
        return {"error": "No session."}

    session = json.load(open(USER_SESSION_FILE))
    base_doc_id = session.get("documentId")
    if not base_doc_id:
        return {"error": "No document ID."}

       # Try in-memory cache first
    if base_doc_id not in FAISS_CACHE:
        index_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_index.faiss")
        sentences_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_sentences.json")
        if os.path.exists(index_path) and os.path.exists(sentences_path):
            index = faiss.read_index(index_path)
            with open(sentences_path, "r") as f:
                sentence_page_map = json.load(f)
            FAISS_CACHE[base_doc_id] = {
                "index": index,
                "sentences": sentence_page_map
            }
        else:
            return {"error": "No FAISS index found. Please upload again."}

    tokenizer = PunktSentenceTokenizer()
    claims = tokenizer.tokenize(request.statement)
    embed = OllamaEmbeddings(model="nomic-embed-text")
    claim_vectors = embed.embed_documents(claims)
    claim_matrix = np.array(claim_vectors).astype("float32")
    faiss.normalize_L2(claim_matrix)

    index_data = FAISS_CACHE[base_doc_id]
    index = index_data["index"]
    sentence_page_map = index_data["sentences"]

    D, I = index.search(claim_matrix, 1)
    results = []

    for i, claim in enumerate(claims):
        best_idx = I[i][0]
        best_score = float(D[i][0])
        sentence, page = sentence_page_map[best_idx]
        results.append({
            "claim": claim,
            "score": best_score,
            "status": "supported" if best_score >= 0.7 else "unsupported",
            "evidence": sentence if best_score >= 0.7 else None,
            "page": page if best_score >= 0.7 else None
        })

    return {"fact_check": results}