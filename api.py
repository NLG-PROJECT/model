from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
import asyncio
from utils.models import ChatRequest, ChatResponse, FactCheckRequest
from utils.constants import DOCUMENTS_DIR, OCR_OUTPUT_DIR
from utils.utilities import log_user_event, clear_user_logs
from experimental import (
    chat,
    get_market_summary,
    get_risk_factors,
    generate_market_summary,
    generate_risk_factors
)
from base import upload_files
from fact_checking import fact_check
from fastapi.middleware.cors import CORSMiddleware

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Models


# API Endpoints
@app.post("/upload/files")
async def upload_files_endpoint(files: List[UploadFile] = File(...)):
    try:
        logger.info("Received file upload request")
        # Log the files we received
        for file in files:
            logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Clear logs first
        logger.info("Clearing user logs...")
        await clear_user_logs()
        logger.info("User logs cleared successfully")
        
        # Process files
        return await upload_files(files)
    except Exception as e:
        logger.error(f"Error in upload_files endpoint: {str(e)}")
        await log_user_event("upload_files", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat endpoint."""
    try:
        return await chat(request)
    except Exception as e:
        log_user_event("chat", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fact-check")
async def fact_check_endpoint(request: FactCheckRequest) -> Dict[str, Any]:
    """Fact check endpoint."""
    try:
        return await fact_check(request)
    except Exception as e:
        log_user_event("fact_check", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-summary")
async def market_summary_endpoint() -> Dict[str, Any]:
    """Market summary endpoint."""
    try:
        return await get_market_summary()
    except Exception as e:
        log_user_event("market_summary", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk-factors")
async def risk_factors_endpoint() -> Dict[str, Any]:
    """Risk factors endpoint."""
    try:
        return await get_risk_factors()
    except Exception as e:
        log_user_event("risk_factors", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-market-summary")
async def generate_market_summary_endpoint() -> Dict[str, Any]:
    """Generate market summary endpoint."""
    try:
        return await generate_market_summary()
    except Exception as e:
        log_user_event("generate_market_summary", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-risk-factors")
async def generate_risk_factors_endpoint() -> Dict[str, Any]:
    """Generate risk factors endpoint."""
    try:
        return await generate_risk_factors()
    except Exception as e:
        log_user_event("generate_risk_factors", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/market")
def get_market_summary():
    try:
        return get_market_summary()
    except Exception as e:
        log_user_event("get_market_summary", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/risk")
def get_risk_factors():
    try:
        return get_risk_factors()
    except Exception as e:
        log_user_event("get_risk_factors", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/sentences")
def debug_list_indexed_sentences(doc_id: str):
    # Implementation for debug sentences
    pass

@app.get("/debug/index-status")
def debug_index_status(doc_id: str):
    # Implementation for debug index status
    pass

@app.get("/debug/schema")
def debug_schema():
    # Implementation for debug schema
    pass

@app.post("/debug/query-sample")
def debug_query_sample(query_text: str):
    # Implementation for debug query sample
    pass

@app.get("/debug/redis-vector")
def debug_redis_vector(doc_id: str, sent_index: int = 0):
    # Implementation for debug redis vector
    pass

@app.get("/debug/list-doc-keys")
def debug_list_doc_keys(doc_id: str):
    # Implementation for debug list doc keys
    pass

@app.get("/debug/sentence-vector")
def debug_sentence_vector(text: str):
    # Implementation for debug sentence vector
    pass

@app.get("/debug/faiss-status")
def debug_faiss_status():
    # Implementation for debug faiss status
    pass

@app.get("/debug/faiss-sentences")
def debug_faiss_sentences(doc_id: str):
    # Implementation for debug faiss sentences
    pass

@app.get("/")
def root():
    # Implementation for root
    pass

@app.post("/old-fact-check")
def old_fact_check(request: FactCheckRequest):
    # Implementation for old fact check
    pass

@app.get("/financial-statements")
async def get_financial_statements() -> Dict[str, Any]:
    """Get financial statements from the clean_output.json file."""
    try:
        log_user_event("financial_statements", "fetch_started")
        json_path = os.path.join(OCR_OUTPUT_DIR, "clean_output.json")
        
        if not os.path.exists(json_path):
            raise HTTPException(
                status_code=404,
                detail="Financial statements not found. Please upload a document first."
            )
            
        with open(json_path, 'r') as f:
            statements = json.load(f)
            
        log_user_event("financial_statements", "fetch_completed")
        return {
            "message": "Financial statements retrieved successfully.",
            "statements": statements
        }
        
    except Exception as e:
        log_user_event("financial_statements", "fetch_error", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 