import os
from typing import List, Dict, Any
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# AskYourPDF and Groq configuration
ASKYOURPDF_API_KEY = os.getenv("ASKYOURPDF_API_KEY")
GROQ_CLOUD_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
ASKYOURPDF_BASE_URL = "https://api.askyourpdf.com/v1/api"

document_ids = []  # Global list for demo purposes (you can use Redis or DB)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 1. Upload file to AskYourPDF
@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    headers = {"x-api-key": ASKYOURPDF_API_KEY}
    new_document_ids = []

    for file in files:
        response = requests.post(
            f"{ASKYOURPDF_BASE_URL}/upload_pdf",
            headers=headers,
            files={"file": (file.filename, file.file, file.content_type)}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="AskYourPDF upload failed")

        doc_id = response.json().get("document_id")
        if not doc_id:
            raise HTTPException(status_code=500, detail="No document ID returned")

        new_document_ids.append(doc_id)

    document_ids.extend(new_document_ids)
    return {"message": "Upload successful", "document_ids": new_document_ids}

# 2. Search AskYourPDF and use Groq for response
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not document_ids:
        raise HTTPException(status_code=400, detail="No uploaded documents available")

    # Query AskYourPDF
    headers = {"x-api-key": ASKYOURPDF_API_KEY}
    params = {
        "query": request.message,
        "document_ids": document_ids
    }
    response = requests.get(f"{ASKYOURPDF_BASE_URL}/search_documents", headers=headers, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="AskYourPDF search failed")

    results = response.json().get("results", [])
    context_chunks = [r.get("text", "") for r in results if "text" in r]
    context = "\n---\n".join(context_chunks)

    if not context:
        return ChatResponse(response="No relevant chunks found.")

    # Query Groq
    try:
        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        prompt = (
            f"You are an AI assistant helping with document Q&A.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.message}\n\n"
            f"Answer concisely with accurate and well-structured information."
        )

        messages = [
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model="llama-3-70b-8192",
            messages=messages,
            temperature=0.6,
            max_tokens=1024,
            top_p=0.9
        )

        return ChatResponse(response=completion.choices[0].message.content.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq response error: {str(e)}")

@app.get("/")
def root():
    return {"message": "AskYourPDF + Groq AI API is running"}
