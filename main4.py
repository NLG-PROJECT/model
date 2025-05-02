# sec_filing_chatbot.py (Ollama + Groq version)

import os
import uuid
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import fitz  # PyMuPDF
import logging
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain_ollama import OllamaEmbeddings
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
OLLAMA_MODEL = "nomic-embed-text"

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_INDEX = "sec_chunks"
EMBEDDING_DIM = 768

# FastAPI setup
app = FastAPI(title="SEC Filing Chatbot (Ollama + Groq)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_filing")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Ensure Redis index
try:
    redis_client.ft(REDIS_INDEX).info()
except:
    schema = (
        TextField("chunk"),
        TextField("section"),
        VectorField("embedding", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": EMBEDDING_DIM,
            "DISTANCE_METRIC": "COSINE"
        })
    )
    definition = IndexDefinition(prefix=["sec:"], index_type=IndexType.HASH)
    redis_client.ft(REDIS_INDEX).create_index(schema, definition=definition)

# Models
class ChatRequest(BaseModel):
    question: str

# Utilities
def extract_text_from_pdf(filepath: str) -> List[str]:
    doc = fitz.open(filepath)
    texts = []
    for page in doc:
        text = page.get_text()
        if text:
            texts.append(text)
    return texts

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    return chunks

def embed_texts(chunks: List[str]) -> List[np.ndarray]:
    try:
        embeddings_model = OllamaEmbeddings(model=OLLAMA_MODEL)
        chunk_embeddings = embeddings_model.embed_documents(chunks)
        logger.debug(f"Generated {len(chunk_embeddings)} embeddings. Example shape: {len(chunk_embeddings[0]) if chunk_embeddings else 'N/A'}")
        return [np.array(vec, dtype=np.float32) for vec in chunk_embeddings]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed.")

def save_chunks_to_redis(chunks: List[str], embeddings: List[np.ndarray], source: str):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"sec:{uuid.uuid4().hex}"
        redis_client.hset(doc_id, mapping={
            "chunk": chunk,
            "section": source,
            "embedding": embedding.tobytes()
        })
        logger.debug(f"Saved chunk {i+1}/{len(chunks)} to Redis with ID {doc_id}")

def search_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    try:
        embeddings_model = OllamaEmbeddings(model=OLLAMA_MODEL)
        query_vec = embeddings_model.embed_query(query)
        query_np = np.array(query_vec, dtype=np.float32)
        redis_query = (
            Query(f"*=>[KNN {top_k} @embedding $vector AS score]")
            .sort_by("score")
            .return_fields("chunk", "score", "section")
            .paging(0, top_k)
            .dialect(2)
        )
        params = {"vector": query_np.tobytes()}
        results = redis_client.ft(REDIS_INDEX).search(redis_query, params)
        logger.debug(f"Search returned {results.total} results for query: '{query}'")
        for i, doc in enumerate(results.docs):
            logger.debug(f"Result {i+1}: Score={doc.score}, Section={doc.section[:30]}, Chunk Preview={doc.chunk[:100]}")
        return [doc.chunk for doc in results.docs]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

def ask_llm(question: str, context: str) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"You are a financial assistant. Use the context from an SEC filing to answer the question.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        answer = ""
        for chunk in response:
            content = getattr(chunk.choices[0].delta, 'content', '')
            answer += content if content else ''
        return answer.strip() or "No valid response generated."
    except Exception as e:
        logger.error(f"Groq LLM failed: {e}")
        raise HTTPException(status_code=500, detail="LLM response generation failed.")

# Endpoints
@app.post("/upload")
async def upload_sec_filing(file: UploadFile = File(...)):
    filepath = f"temp_{uuid.uuid4().hex}.pdf"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        all_text = extract_text_from_pdf(filepath)
        logger.info(f"Extracted {len(all_text)} pages from PDF.")
        all_chunks = []
        for section_text in all_text:
            chunks = chunk_text(section_text)
            logger.debug(f"Extracted {len(chunks)} chunks from one section.")
            all_chunks.extend(chunks)
        if not all_chunks:
            logger.warning("No valid chunks extracted from the document.")
            raise HTTPException(status_code=400, detail="No text found in the uploaded file.")
        embeddings = embed_texts(all_chunks)
        save_chunks_to_redis(all_chunks, embeddings, source=file.filename)
        logger.info(f"Stored {len(all_chunks)} chunks with embeddings in Redis.")
        return {"message": "File processed", "chunks_stored": len(all_chunks)}
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(filepath)

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received chat question: {request.question}")
    chunks = search_similar_chunks(request.question)
    if not chunks:
        logger.warning("No similar chunks found for the question.")
        return {"answer": "I couldn't find relevant information in the filings."}
    context = "\n\n".join(chunks)
    answer = ask_llm(request.question, context)
    logger.info("LLM generated a response.")
    return {"answer": answer}

@app.get("/")
def health_check():
    return {"message": "SEC Filing Chatbot (Ollama + Groq) is running."}
