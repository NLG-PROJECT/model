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

# Caches FAISS index to user session for fact checking
FAISS_CACHE = {}

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Explicitly load and assign the Punkt tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Create documents directory if it doesn't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Redis init
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)

def ensure_redis_index():
    try:
        redis_client.ft(REDIS_INDEX_NAME).info()
    except:
        schema = (
            TextField("chunk"),
            TextField("source"),
            NumericField("page"),
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIMENSION,
                "DISTANCE_METRIC": "COSINE"
            })
        )
        redis_client.ft(REDIS_INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
        )
        print("✅ Redis vector index created")

ensure_redis_index()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


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
        print(chunks, "chunks")
        log_user_event("extract_text", "completed", f"{len(chunks)} chunks")

        # ✅ Save raw content for fact checking
        raw_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as rf:
            for text, page in content:
                rf.write(f"[PAGE {page}]\n{text}\n\n")

        async def embed_and_store():
            try:
                log_user_event("embedding", "started")
                embeddings = embed_text_chunks(chunks)
                add_embeddings_to_redis(embeddings, chunks, base_doc_id, file.filename)
                log_user_event("embedding", "completed")
            except Exception as e:
                log_user_event("embedding", "error", str(e))

        # Sentence indexing background process with logging

        async def upload_to_askyourpdf():
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
            return ask_doc_id

              # FAISS Sentence Indexing (replaces Redis indexing)
        async def index_with_faiss():
            log_user_event("faiss_indexing", "started")
            sentence_page_map = []
            for text, page in content:
                for sentence in punkt_tokenizer.tokenize(text):
                    sentence_page_map.append((sentence.strip(), page))

            if not sentence_page_map:
                raise ValueError("No sentences found for FAISS indexing")

            sentences = [s for s, _ in sentence_page_map]
            embed = OllamaEmbeddings(model="nomic-embed-text")
            vectors = embed.embed_documents(sentences)
            matrix = np.array(vectors).astype("float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(len(matrix[0]))
            index.add(matrix)

            # Cache in memory
            FAISS_CACHE[base_doc_id] = {
                "index": index,
                "sentences": sentence_page_map
            }
             
            index_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_index.faiss")
            faiss.write_index(index, index_path)

            # Save sentence map for fallback
            sentence_map_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_sentences.json")
            with open(sentence_map_path, "w") as f:
                json.dump(sentence_page_map, f, indent=2)


            log_user_event("faiss_indexing", "completed", f"{len(sentences)} sentences")

        metadata_entry = {
            "documentId": base_doc_id,
            "documentName": file.filename,
            "askyourpdfdocId": None
        }

        update_files_list(metadata_entry)
        reset_user_session(metadata_entry)
        log_user_event("session_reset", "completed")
        task_embed = asyncio.create_task(embed_and_store())
        task_ask = asyncio.create_task(upload_to_askyourpdf())
        task_index = asyncio.create_task(index_with_faiss())

        ask_doc_id = await task_ask
        await asyncio.gather(task_embed, task_index)
        if ask_doc_id:
            session_data = json.load(open(USER_SESSION_FILE))
            session_data["askyourpdfdocId"] = ask_doc_id
            with open(USER_SESSION_FILE, "w") as f:
                json.dump(session_data, f, indent=2)

        return {"message": "File processed successfully.", "documentId": base_doc_id}

    results = await asyncio.gather(*[process_file(file) for file in files])
    return {"message": f"Processed {len(results)} files.", "results": results}



