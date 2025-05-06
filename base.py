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
import aiofiles
import aiohttp

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
    try:
        # Clear logs first, before anything else
        logger.info("Starting upload process - clearing logs first")
        await clear_user_logs()
        logger.info("Logs cleared successfully")
        
        # Start batch logging
        await log_user_event("upload_batch", "start", f"{len(files)} files")
        logger.info(f"Starting upload of {len(files)} files...")

        async def process_file(file: UploadFile):
            try:
                await log_user_event("upload_start", "started", file.filename)
                logger.info(f"Processing file: {file.filename}")

                # Group 1: File saving and initial processing
                file_bytes = await file.read()
                ext = file.filename.split(".")[-1]
                filename = f"{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(DOCUMENTS_DIR, filename)
                
                logger.info(f"Saving file to: {filepath}")
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(file_bytes)

                base_doc_id = f"doc:{uuid.uuid4().hex}"
                logger.info(f"Generated doc_id: {base_doc_id}")

                await log_user_event("extract_text", "started", file.filename)
                logger.info("Starting text extraction...")
                content = await extract_text_from_file(filepath)
                chunks = chunk_text(content)  # This can stay sync as it's pure computation
                logger.info(f"Extracted {len(chunks)} chunks")
                await log_user_event("extract_text", "completed", f"{len(chunks)} chunks")

                # Save raw content for fact checking
                raw_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_raw.txt")
                logger.info(f"Saving raw content to: {raw_path}")
                async with aiofiles.open(raw_path, "w", encoding="utf-8") as rf:
                    for text, page in content:
                        await rf.write(f"[PAGE {page}]\n{text}\n\n")

                # Initialize metadata and reset session
                metadata_entry = {
                    "documentId": base_doc_id,
                    "documentName": file.filename,
                    "askyourpdfdocId": None
                }

                logger.info("Updating files list and resetting session...")
                await update_files_list(metadata_entry)
                await reset_user_session(metadata_entry)
                await log_user_event("session_reset", "completed")
                logger.info("Session reset completed")

                # Create all tasks
                logger.info("Creating parallel tasks...")
                tasks = [
                    embed_and_store(chunks, base_doc_id, file.filename),
                    index_with_faiss(content, base_doc_id),
                    process_financial_statements(filepath),
                    upload_to_askyourpdf(file_bytes, file.filename, file.content_type)
                ]

                # Wait for all tasks to complete
                logger.info("Waiting for all tasks to complete...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Get ask_doc_id from the last result (upload_to_askyourpdf)
                ask_doc_id = results[-1] if not isinstance(results[-1], Exception) else None
                logger.info(f"AskYourPDF upload completed with doc_id: {ask_doc_id}")

                if ask_doc_id:
                    logger.info("Updating session with AskYourPDF doc_id...")
                    async with aiofiles.open(USER_SESSION_FILE, "r") as f:
                        session_data = json.loads(await f.read())
                    session_data["askyourpdfdocId"] = ask_doc_id
                    async with aiofiles.open(USER_SESSION_FILE, "w") as f:
                        await f.write(json.dumps(session_data, indent=2))
                    logger.info("Session updated with AskYourPDF doc_id")

                # Check for any exceptions in the results
                for i, result in enumerate(results[:-1]):  # Skip the last result (ask_doc_id)
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed with error: {str(result)}")
                        raise result

                return {"message": "File processed successfully.", "documentId": base_doc_id}
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                await log_user_event("file_processing", "error", str(e))
                raise

        async def upload_to_askyourpdf(file_bytes, filename, content_type):
            ask_doc_id = None
            await log_user_event("askyourpdf_upload", "started")
            try:
                async with aiohttp.ClientSession() as session:
                    # Create a proper multipart form data
                    data = aiohttp.FormData()
                    data.add_field('file',
                                 file_bytes,
                                 filename=filename,
                                 content_type=content_type)
                    
                    async with session.post(
                        f"{ASKYOURPDF_BASE_URL}/api/upload",
                        headers={"x-api-key": ASKYOURPDF_API_KEY},
                        data=data
                    ) as response:
                        if response.status == 201:
                            result = await response.json()
                            ask_doc_id = result.get("docId")
                            await log_user_event("askyourpdf_upload", "completed", ask_doc_id)
                        else:
                            error_text = await response.text()
                            logger.error(f"AskYourPDF upload failed: {error_text}")
                            await log_user_event("askyourpdf_upload", "failed", error_text)
            except Exception as e:
                logger.error(f"Error in AskYourPDF upload: {str(e)}")
                await log_user_event("askyourpdf_upload", "error", str(e))
            return ask_doc_id

        async def embed_and_store(chunks, base_doc_id, filename):
            try:
                await log_user_event("embedding", "started")
                embeddings = await embed_text_chunks(chunks)
                await add_embeddings_to_redis(embeddings, chunks, base_doc_id, filename)
                await log_user_event("embedding", "completed")
            except Exception as e:
                await log_user_event("embedding", "error", str(e))

        async def process_financial_statements(filepath):
            try:
                await log_user_event("financial_statements", "started")
                from factual import obtain_financial_statements, locate_financial_pages
                
                # First locate the pages and await the result
                pages = await locate_financial_pages(filepath)
                logger.info(f"Located financial pages: {pages}")
                
                # Then get the statements and await the result
                statements = await obtain_financial_statements(filepath)
                logger.info("Obtained financial statements")
                
                # Update session with the statements
                async with aiofiles.open(USER_SESSION_FILE, "r") as f:
                    session_data = json.loads(await f.read())
                session_data["financial_statements"] = statements
                async with aiofiles.open(USER_SESSION_FILE, "w") as f:
                    await f.write(json.dumps(session_data, indent=2))
                
                await log_user_event("financial_statements", "completed")
                return statements  # Return the statements for potential use
            except Exception as e:
                logger.error(f"Error in process_financial_statements: {str(e)}")
                await log_user_event("financial_statements", "error", str(e))
                raise

        async def index_with_faiss(content, base_doc_id):
            await log_user_event("faiss_indexing", "started")
            sentence_page_map = []
            for text, page in content:
                for sentence in punkt_tokenizer.tokenize(text):
                    sentence_page_map.append((sentence.strip(), page))

            if not sentence_page_map:
                raise ValueError("No sentences found for FAISS indexing")

            sentences = [s for s, _ in sentence_page_map]
            embed = OllamaEmbeddings(model="nomic-embed-text")
            vectors = await embed.embed_documents(sentences)
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
            async with aiofiles.open(sentence_map_path, "w") as f:
                await f.write(json.dumps(sentence_page_map, indent=2))

            await log_user_event("faiss_indexing", "completed", f"{len(sentences)} sentences")

        # Process all files in parallel
        logger.info("Starting parallel file processing...")
        results = await asyncio.gather(*[process_file(file) for file in files])
        logger.info(f"Successfully processed {len(results)} files")
        return {"message": f"Processed {len(results)} files.", "results": results}
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        await log_user_event("upload_batch", "error", str(e))
        raise



