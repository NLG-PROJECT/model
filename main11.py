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

# Caches FAISS index to user session for fact checking
FAISS_CACHE = {}

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
SECONDARY_INDEX_NAME = "sentences_idx"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Download punkt if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Explicitly load and assign the Punkt tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

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



# Models
class ChatRequest(BaseModel):
    message: str
    option: str
    class Config:
        extra = "forbid"

class ChatResponse(BaseModel):
    response: str

class FactCheckRequest(BaseModel):
    statement: str


# Utils
### TEXT EXTRACTION (Parallel with ThreadPool) <<< UPDATED >>>
### TEXT EXTRACTION (Parallel with ThreadPool + page number logging) <<< UPDATED >>>
def extract_text_from_file(filepath: str) -> List[tuple[str, int]]:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return [(page.extract_text(), i + 1) for i, page in enumerate(reader.pages) if page.extract_text()]
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return [(p.text, 1) for p in doc.paragraphs if p.text.strip()]
    else:
        raise ValueError("Unsupported file format")
### EMBEDDING UTILS

def chunk_text(pages: list[tuple[str, int]], chunk_size: int = 500, overlap: int = 50) -> List[tuple[str, int]]:
    chunks = []
    for text, page in pages:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append((chunk, page))
    return chunks

def embed_text_chunks(text_chunks: list[tuple[str, int]]) -> List[np.ndarray]:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    texts_only = [chunk for chunk, _ in text_chunks]
    return [np.array(e).astype('float32') for e in embeddings.embed_documents(texts_only)]

def add_embeddings_to_redis(embeddings: list[np.ndarray], chunks: List[tuple[str, int]], base_doc_id: str, source: str):
    for i, ((chunk, page), embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{base_doc_id}:chunk:{i}"
        redis_client.hset(doc_id, mapping={
            "chunk": chunk,
            "page": page,
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


# def retrieve_similar_chunks(prompt: str, top_k: int = 5) -> List[str]:
#     try:
#         embeddings = OllamaEmbeddings(model="nomic-embed-text")
#         query_embedding = np.array(embeddings.embed_query(prompt)).astype('float32')
#         query = f"*=>[KNN {top_k} @embedding $vector AS score]"
#         params_dict = {"vector": query_embedding.tobytes()}
#         redis_query = redis_client.ft(REDIS_INDEX_NAME).search(
#             redis.commands.search.query.Query(query).dialect(2).sort_by("score").paging(0, top_k).return_fields("chunk", "score"),
#             params_dict
#         )
#         results = redis_query.docs
#         print(results)
#         good_chunks = [doc.chunk for doc in results if float(doc.score) <= SIMILARITY_THRESHOLD]
#         return good_chunks
#     except Exception as e:
#         log_user_event("redis_search", "error", str(e))
#         return []

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
            fallback_chunks = chunk_text([(context, -1)])
            fallback_embeddings = embed_text_chunks(fallback_chunks)
            fallback_base = f"{base_doc_id}:askfallback"
            add_embeddings_to_redis(fallback_embeddings, fallback_chunks, fallback_base, source="askyourpdf_chat_fallback")

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

# MAIN COMBINED ROUTE
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

        return len(chunks)

    tasks = [process_file(file) for file in files]
    chunk_counts = await asyncio.gather(*tasks)
    total_chunks = sum(chunk_counts)

    log_user_event("upload_batch", "complete", f"{total_chunks} chunks embedded")
    return {
        "message": "Files processed for both Redis and AskYourPDF.",
        "embedded_chunks_count": total_chunks
    }


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


@app.post("/generate/market-summary")
def generate_market_summary():
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


@app.post("/generate/risk-factors")
def generate_risk_factors():
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


@app.get("/summary/market")
def get_market_summary():
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


@app.get("/summary/risk")
def get_risk_factors():
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



### FACT CHECKER

def old_fact_check_answer(statement: str, threshold: float = 0.7) -> Dict[str, Any]:
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


@app.get("/debug/sentences")
def debug_list_indexed_sentences(doc_id: str):
    import logging
    logger = logging.getLogger("FactCheck")
    try:
        keys = redis_client.keys(f"{doc_id}:sent:*")
        sentences = []
        for key in keys:
            data = redis_client.hgetall(key)
            sentence = data.get(b"sentence", b"").decode("utf-8")
            page = data.get(b"page", b"").decode("utf-8")
            sentences.append({"key": key.decode(), "sentence": sentence, "page": page})
        logger.info(f"Retrieved {len(sentences)} indexed sentences for {doc_id}.")
        return {"indexed_sentences": sentences}
    except Exception as e:
        logger.error(f"Debug retrieval failed for {doc_id}: {e}")
        return {"error": str(e)}


@app.get("/debug/index-status")
def debug_index_status(doc_id: str):
    status = redis_client.get(f"{doc_id}:sentences_indexed")
    return {"doc_id": doc_id, "status": status.decode("utf-8") if status else None}

@app.get("/debug/schema")
def debug_schema():
    try:
        info = redis_client.ft(SECONDARY_INDEX_NAME).info()
        return {"index": SECONDARY_INDEX_NAME, "fields": info["attributes"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/query-sample")
def debug_query_sample(query_text: str):
    from langchain_ollama import OllamaEmbeddings
    import numpy as np
    import logging
    logger = logging.getLogger("FactCheck")

    try:
        embed = OllamaEmbeddings(model="nomic-embed-text")
        vector = embed.embed_documents([query_text])[0]  # UPDATED: consistent embedding method
        np_vector = np.array(vector, dtype='float32')
        logger.info(f"Query vector: dim={len(vector)}, norm={np.linalg.norm(np_vector):.4f}")

        query = f"*=>[KNN 5 @embedding $vector AS score]"
        params_dict = {"vector": np_vector.tobytes()}

        redis_query = redis_client.ft(SECONDARY_INDEX_NAME).search(
            redis.commands.search.query.Query(query)
            .dialect(2)
            .paging(0, 5)
            .return_fields("sentence", "page", "score"),
            params_dict
        )

        results = []
        for doc in redis_query.docs:
            results.append({
                "sentence": getattr(doc, "sentence", None),
                "page": getattr(doc, "page", None),
                "score": float(doc.score)
            })

        if not results:
            logger.warning("No vector results returned for query.")

        return {
            "query_vector_first_values": vector[:10],
            "query_vector_norm": float(np.linalg.norm(np_vector)),
            "results": results
        }
    except Exception as e:
        logger.error(f"Query sample error: {e}")
        return {"error": str(e)}


@app.get("/debug/redis-vector")
def debug_redis_vector(doc_id: str, sent_index: int = 0):
    import numpy as np
    key = f"{doc_id}:sent:{sent_index}"
    data = redis_client.hgetall(key)
    vec_bytes = data.get(b"embedding", None)
    if not vec_bytes:
        return {"error": f"No vector found at {key}"}
    vec = np.frombuffer(vec_bytes, dtype=np.float32)
    return {
        "key": key,
        "dim": vec.shape[0],
        "first_values": vec[:10].tolist(),
        "norm": float(np.linalg.norm(vec)),
        "sum": float(np.sum(vec))
    }

@app.get("/debug/list-doc-keys")
def debug_list_doc_keys(doc_id: str):
    try:
        keys = redis_client.keys(f"{doc_id}:sent:*")
        return {"keys": [key.decode() for key in keys]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/sentence-vector")
def debug_sentence_vector(text: str):
    from langchain_ollama import OllamaEmbeddings
    import numpy as np
    try:
        embed = OllamaEmbeddings(model="nomic-embed-text")
        vec = embed.embed_query(text)
        return {
            "dim": len(vec),
            "first_values": vec[:10],
            "sum": float(np.sum(vec)),
            "norm": float(np.linalg.norm(vec))
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/fact-check")
def fact_check(request: FactCheckRequest):
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


@app.get("/debug/faiss-status")
def debug_faiss_status():
    return {"cached_docs": list(FAISS_CACHE.keys())}

@app.get("/debug/faiss-sentences")
def debug_faiss_sentences(doc_id: str):
    if doc_id not in FAISS_CACHE:
        return {"error": "Not found in FAISS cache."}
    return {"sentences": FAISS_CACHE[doc_id]["sentences"][:5]}  # preview first 5


def save_faiss_to_session(sentence_page_map: List[tuple[str, int]], dim: int):
    session_data = json.load(open(USER_SESSION_FILE))
    session_data["faiss_sentences"] = sentence_page_map
    session_data["faiss_dim"] = dim
    with open(USER_SESSION_FILE, "w") as f:
        json.dump(session_data, f, indent=2)
        
def index_sentences_for_fact_checking(pages: List[tuple[str, int]], base_doc_id: str):
    import logging
    logger = logging.getLogger("FactCheck")
    log_user_event("sentence_indexing", "started")
    try:
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()

        logger.info(f"Starting sentence indexing for {base_doc_id} with {len(pages)} pages.")

        all_sentences = []
        for i, (text, page) in enumerate(pages):
            if not text.strip():
                logger.debug(f"Page {page} is empty. Skipping.")
                continue
            try:
                sentences = tokenizer.tokenize(text)
                logger.info(f"Page {page} has {len(sentences)} sentences.")
                all_sentences.extend([(s, page) for s in sentences if s.strip()])
            except Exception as e:
                logger.error(f"Tokenization failed on page {page}: {e}")

        if not all_sentences:
            logger.warning("No sentences found after tokenization.")
            redis_client.set(f"{base_doc_id}:sentences_indexed", "error")
            return

        texts_only = [s for s, _ in all_sentences]
        logger.info(f"Total sentences to index: {len(texts_only)}")

        embed = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = [np.array(e).astype('float32') for e in embed.embed_documents(texts_only)]
        logger.info("Embeddings generated for all sentences.")

        success_count = 0
        for i, ((sentence, page), embedding) in enumerate(zip(all_sentences, embeddings)):
            try:
                doc_id = f"{base_doc_id}:sent:{i}"
                redis_client.hset(doc_id, mapping={
                    "sentence": sentence,
                    "page": page,
                    "embedding": embedding.tobytes()
                })
                success_count += 1
            except Exception as e:
                logger.error(f"Redis insertion failed for sentence {i} on page {page}: {e}")

        redis_client.set(f"{base_doc_id}:sentences_indexed", "success")
        logger.info(f"Sentence indexing completed successfully. {success_count}/{len(all_sentences)} sentences indexed.")
        log_user_event("sentence_indexing", "completed")
    except Exception as e:
        redis_client.set(f"{base_doc_id}:sentences_indexed", "error")
        logger.error(f"Sentence indexing failed: {e}")
        log_user_event("sentence_indexing", "error", str(e))

@app.get("/")
def root():
    return {"message": "DocuGroq server running"}



@app.post("/old-fact-check")
def old_fact_check(request: FactCheckRequest):
    return old_fact_check_answer(request.statement)

@app.get("/")
def root():
    return {"message": "Server is running"}




ASK_YOUR_SUMMARY_PROMPT = """
You are FinGroq, a financial analyst AI used to produce structured, fluent, and data-driven market summaries from SEC filings.

Your goal is to generate a summary with the following sections (if available):  
1. Industry Overview  
2. Competitive Landscape  
3. Customer Segments  
4. Revenue Breakdown  
5. Risks and Challenges  
6. Geographic Exposure  
7. Macroeconomic Sensitivity  
8. Innovation and R&D  
9. Strategic Initiatives  
10. ESG and Legal Disclosures

Follow this format:
- Use markdown-style headers (###) for each section.
- Write concise, fact-based paragraphs (not bullets).
- Include figures, percentages, and dollar values.
- Omit any section if content is not available.
- Do not add opinions or interpretations.

Example:

### Industry Overview  
Intel operates in the semiconductor industry, producing microprocessors, chipsets, and SoCs.  
In 2023, industry growth was driven by AI, edge infrastructure, and demand for customizable chip systems.  
Risks included high R&D costs and geopolitical instability such as U.S.–China tensions.

### Competitive Landscape  
Intel competes with AMD, NVIDIA, TSMC, and Qualcomm.  
The company’s data center market share declined due to ARM-based solutions and GPU acceleration trends.

Use this format and tone. Base your output solely on the document.
"""

ASK_RISK_FACTORS_SUMMARY_PROMPT = """
You are FinGroq, a financial analyst AI. Your task is to summarize the material risks disclosed in the Risk Factors section of an SEC filing.

Instructions:
- Use only the content in the document. Do not add, assume, or infer anything.
- Structure the summary using these categories if supported:
  ### Market Risks
  ### Geopolitical & Macroeconomic Risks
  ### Operational Risks
  ### AI & Technology Risks
  ### Cybersecurity & IP Risks
  ### Legal & Regulatory Risks
  ### Environmental Risks
- Write fluent, concise paragraphs under each heading.
- Include figures, country names, or impacts when mentioned.
- Omit any category not directly supported by the document.

Example:

### Market Risks  
Intel faces growing competition from ARM-based chips and GPUs, which threaten market share across data center and client segments.

### Geopolitical & Macroeconomic Risks  
Tensions with China, the war in Ukraine, and instability in the Middle East affect trade policy, supply chains, and raise cyberattack risks.

### Operational Risks  
The IDM 2.0 strategy requires major capital investment. With $50B in debt, delays in foundry scaling may reduce returns.

### Cybersecurity & IP Risks  
Intel is a frequent target of state-sponsored threats. Breaches risk IP theft, legal exposure, and operational disruption.

Use this format and tone. Base your output solely on the document.
"""
