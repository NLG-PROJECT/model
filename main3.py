# sec_filing_chatbot.py

import os
import uuid
import openai
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

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_INDEX = "sec_chunks"
EMBEDDING_DIM = 1536

# FastAPI setup
app = FastAPI(title="SEC Filing Chatbot")
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
    response = openai.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [np.array(e.embedding, dtype=np.float32) for e in response.data]


def save_chunks_to_redis(chunks: List[str], embeddings: List[np.ndarray], source: str):
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = f"sec:{uuid.uuid4().hex}"
        redis_client.hset(doc_id, mapping={
            "chunk": chunk,
            "section": source,
            "embedding": embedding.tobytes()
        })


def search_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    query_vec = openai.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding
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
    return [doc.chunk for doc in results.docs]


def ask_llm(question: str, context: str) -> str:
    prompt = (
        f"You are a financial assistant. Use the context from an SEC filing to answer the question.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    completion = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    return completion.choices[0].message.content.strip()


# Endpoints
@app.post("/upload")
async def upload_sec_filing(file: UploadFile = File(...)):
    filepath = f"temp_{uuid.uuid4().hex}.pdf"
    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        all_text = extract_text_from_pdf(filepath)
        all_chunks = []
        for section_text in all_text:
            chunks = chunk_text(section_text)
            all_chunks.extend(chunks)

        embeddings = embed_texts(all_chunks)
        save_chunks_to_redis(all_chunks, embeddings, source=file.filename)

        return {"message": "File processed", "chunks_stored": len(all_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(filepath)


@app.post("/chat")
async def chat(request: ChatRequest):
    chunks = search_similar_chunks(request.question)
    if not chunks:
        return {"answer": "I couldn't find relevant information in the filings."}

    context = "\n\n".join(chunks)
    answer = ask_llm(request.question, context)
    return {"answer": answer}


@app.post("/summarize")
async def summarize_sec_filing(file: UploadFile = File(...)):
    filepath = f"temp_{uuid.uuid4().hex}.pdf"
    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        all_text = extract_text_from_pdf(filepath)
        joined_text = "\n\n".join(all_text)[:4000]  # Cap for prompt size

        prompt = f"Summarize the following SEC filing:\n\n{joined_text}"
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        return {"summary": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(filepath)


@app.get("/")
def health_check():
    return {"message": "SEC Filing Chatbot is running."}
