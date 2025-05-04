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
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import pdfplumber
import nltk
from nltk.tokenize import PunktSentenceTokenizer

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

# Download punkt if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Explicitly load and assign the Punkt tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

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
        log_user_event("extract_text", "completed", f"{len(chunks)} chunks")

        # ✅ Save raw content for fact checking
        raw_path = os.path.join(DOCUMENTS_DIR, f"{base_doc_id}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as rf:
            for text, page in content:
                rf.write(f"[PAGE {page}]\n{text}\n\n")

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

        # Run Redis and AskYourPDF in parallel
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
        add_embeddings_to_redis(embeddings, chunks, f"{base_doc_id}:market_summary", annotation="askyourpdf_market_summary")

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
        add_embeddings_to_redis(embeddings, chunks, f"{base_doc_id}:risk_factors", annotation="askyourpdf_risk_factors")

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
    from nltk.tokenize import sent_tokenize
    if not os.path.exists(USER_SESSION_FILE): return {"error": "No session."}
    session = json.load(open(USER_SESSION_FILE))
    doc_id = session.get("documentId")
    if not doc_id: return {"error": "No document ID."}
    raw_file_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}_raw.txt")
    if not os.path.exists(raw_file_path): return {"error": "No raw document."}
    document_text = open(raw_file_path).read()
     
    sentences = punkt_tokenizer.tokenize(document_text)
    claims = punkt_tokenizer.tokenize(statement)# <<< UPDATED >>>
    embed = OllamaEmbeddings(model="nomic-embed-text")
    doc_vectors = embed.embed_documents(sentences)
    claim_vectors = embed.embed_documents(claims)

    results = []
    for i, vec in enumerate(claim_vectors):
        best, best_score = "", 0
        for j, doc_vec in enumerate(doc_vectors):
            score = float(np.dot(vec, doc_vec) / (np.linalg.norm(vec) * np.linalg.norm(doc_vec)))
            if score > best_score:
                best_score = score
                best = sentences[j]
        results.append({
            "claim": claims[i],
            "score": best_score,
            "status": "supported" if best_score >= threshold else "unsupported",
            "evidence": best if best_score >= threshold else None
        })
    return {"fact_check": results}

def fact_check_answer(statement: str, threshold: float = 0.7) -> Dict[str, Any]:
    from nltk.tokenize import sent_tokenize
    if not os.path.exists(USER_SESSION_FILE): return {"error": "No session."}
    session = json.load(open(USER_SESSION_FILE))
    base_doc_id = session.get("documentId")
    if not base_doc_id: return {"error": "No document ID."}

    claims = punkt_tokenizer.tokenize(statement)
    embed = OllamaEmbeddings(model="nomic-embed-text")
    claim_vectors = embed.embed_documents(claims)

    results = []
    for i, vec in enumerate(claim_vectors):
        query = f"*=>[KNN 1 @embedding $vector AS score]"
        params_dict = {"vector": np.array(vec, dtype='float32').tobytes()}

        try:
            redis_query = redis_client.ft(REDIS_INDEX_NAME).search(
                redis.commands.search.query.Query(query).dialect(2).sort_by("score").paging(0, 1).return_fields("chunk", "page", "score"),
                params_dict
            )
            if redis_query.docs:
                doc = redis_query.docs[0]
                score = float(doc.score)
                results.append({
                    "claim": claims[i],
                    "score": score,
                    "status": "supported" if score >= threshold else "unsupported",
                    "evidence": doc.chunk if score >= threshold else None,
                    "page": int(doc.page) if hasattr(doc, 'page') else None
                })
            else:
                results.append({"claim": claims[i], "score": 0.0, "status": "unsupported", "evidence": None})
        except Exception as e:
            results.append({"claim": claims[i], "error": str(e)})

    return {"fact_check": results}

@app.post("/fact-check")
def fact_check(request: FactCheckRequest):
    return fact_check_answer(request.statement)

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
