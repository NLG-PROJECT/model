import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from CQE.chunking.preprocessor import SECFilingPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SEC Filing Summarizer and Chatbot",
    description="API for processing, summarizing, and chatting with SEC filings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and components
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm = HuggingFacePipeline.from_model_id(
    model_id=LLM_MODEL,
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    pipeline_kwargs={"max_new_tokens": 512}
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize vector store
vector_store = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str

class SECFilingMetadata(BaseModel):
    filename: str
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Utility functions
def extract_text_from_file(filepath: str) -> str:
    """Extract text from PDF or DOCX files."""
    logger.info(f"Extracting text from file: {filepath}")
    try:
        if filepath.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filepath.endswith(".docx"):
            doc = Document(filepath)
            text = " ".join([p.text for p in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

def process_sec_filing(text: str) -> Dict[str, Any]:
    """Process SEC filing text using the SECFilingPreprocessor."""
    logger.info("Starting SEC filing preprocessing")
    try:
        preprocessor = SECFilingPreprocessor()
        result = preprocessor.preprocess(text)
        logger.info(f"Processed SEC filing with {len(result['sections'])} sections")
        return result
    except Exception as e:
        logger.error(f"Error preprocessing SEC filing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing SEC filing: {str(e)}")

def generate_summary(text: str) -> str:
    """Generate a summary of the provided text using the LLM."""
    try:
        prompt = f"""Please provide a concise summary of the following SEC filing text, focusing on key points and important information:

{text}

Summary:"""
        response = llm(prompt)
        return response
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# API endpoints
@app.post("/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a SEC filing document."""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract and process the text
        text = extract_text_from_file(file_path)
        processed = process_sec_filing(text)
        
        # Split text into chunks
        chunks = text_splitter.split_text(processed['cleaned_text'])
        
        # Create embeddings and store in FAISS
        global vector_store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks": len(chunks),
            "sections": processed['sections'],
            "metadata": processed['metadata']
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the SEC filing using the LLM."""
    try:
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No SEC filing has been uploaded yet")
        
        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"question": request.question})
        
        # Extract sources
        sources = [doc.page_content for doc in result["source_documents"]]
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    """Generate a summary of the provided text."""
    try:
        summary = generate_summary(request.text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "SEC Filing Summarizer and Chatbot API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 