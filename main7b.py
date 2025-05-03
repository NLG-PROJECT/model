import os
from typing import List, Dict, Any
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from fastapi.routing import APIRoute

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


@app.on_event("startup")
async def list_routes_on_startup():
    print("\n🔍 Available routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            print(f"{route.path} - {route.methods}")
    print(f"📄 Current stored document IDs: {document_ids}\n")


@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    headers = {"x-api-key": ASKYOURPDF_API_KEY}
    new_document_ids = []

    for file in files:
        file_bytes = await file.read()
        print(f"📤 Uploading {file.filename} to AskYourPDF...")

        try:
            response = requests.post(
                f"{ASKYOURPDF_BASE_URL}/upload",
                headers=headers,
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Request to AskYourPDF failed: {str(ex)}")

        print(f"🔁 Status Code: {response.status_code}")
        print(f"📨 Response Text: {response.text}")

        if response.status_code != 201:
            raise HTTPException(status_code=response.status_code, detail=f"AskYourPDF upload failed: {response.text}")

        doc_id = response.json().get("docId")
        if not doc_id:
            raise HTTPException(status_code=500, detail="No document ID returned by AskYourPDF")

        new_document_ids.append(doc_id)

    document_ids.extend(new_document_ids)
    print(f"✅ Uploaded Document IDs: {new_document_ids}")

    return {
        "message": "Upload successful",
        "document_ids": new_document_ids
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # if not document_ids:
    #     raise HTTPException(status_code=400, detail="No uploaded documents available")

    headers = {
        "x-api-key": ASKYOURPDF_API_KEY,
        "Content-Type": "application/json"
    }

    # doc_id = document_ids[-1]  # Use the most recently uploaded document
    doc_id = "a9a14190-83f4-4a29-85b0-24084dedc07e"
    chat_url = f"https://api.askyourpdf.com/v1/chat/{doc_id}"

    messages = [
        {
            "sender": "User",
            "message": request.message
        }
    ]

    try:
        response = requests.post(chat_url, headers=headers, json=messages)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"AskYourPDF chat failed: {response.text}")

        answer = response.json().get("answer", {}).get("message", "")

        if not answer:
            return ChatResponse(response="No answer returned from AskYourPDF.")

        return ChatResponse(response=answer.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with AskYourPDF: {str(e)}")


@app.get("/")
def root():
    return {"message": "AskYourPDF + Groq AI API is running"}


@app.get("/test")
def test():
    return {
        "routes": [route.path for route in app.routes if isinstance(route, APIRoute)],
        "document_ids": document_ids
    }
