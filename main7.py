import os
from typing import List, Dict, Any
import requests
import json
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
ASKYOURPDF_BASE_URL = "https://api.askyourpdf.com/v1"

SESSION_FILE = "user_session.json"
document_ids = []  # Global list for demo purposes

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

    # Initialize chat history session
    with open(SESSION_FILE, "w") as f:
        json.dump({"doc_id": new_document_ids[-1], "chat_history": []}, f)

    return {
        "message": "Upload successful",
        "document_ids": new_document_ids
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    headers = {
        "x-api-key": ASKYOURPDF_API_KEY,
        "Content-Type": "application/json"
    }

    session_data = {}
    doc_id = None
    chat_history = []

    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                session_data = json.load(f)
                doc_id = session_data.get("doc_id")
                chat_history = session_data.get("chat_history", [])
        except Exception as e:
            print(f"⚠️ Error loading session file: {e}")

    if not doc_id:
        if not document_ids:
            raise HTTPException(status_code=400, detail="No uploaded documents available")
        doc_id = document_ids[-1]
        chat_history = []

    # Append new user message to history
    chat_history.append({"sender": "user", "message": request.message})
    chat_url = f"{ASKYOURPDF_BASE_URL}/chat/{doc_id}?stream=False"

    print(f"📤 Sending chat history to AskYourPDF...\nUsing document ID: {doc_id}")

    try:
        response = requests.post(chat_url, headers=headers, json=chat_history)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"AskYourPDF chat failed: {response.text}")

        result = response.json()
        context = result["answer"]["message"]

        # Refined prompt engineering for Groq
        identity = "You are DocuGroq, a world-class AI assistant that provides concise, insightful answers to document-based questions."
        instructions = (
            "Always base your answers solely on the provided context. "
            "Respond in clear, structured paragraphs with well-reasoned logic. "
            "If the context is insufficient, say you don't have enough information."
        )
        prompt = (
            f"{identity}\n\n"
            f"Document Context:\n{context}\n\n"
            f"User Question:\n{request.message}\n\n"
            f"Instructions:\n{instructions}"
        )

        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": prompt},
        ]

        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.55,
            max_tokens=1024,
            top_p=0.85,
            stream=False
        )

        final_response = completion.choices[0].message.content.strip()

        # Update chat history with Groq-enhanced response
        chat_history.append({"sender": "bot", "message": final_response})
        with open(SESSION_FILE, "w") as f:
            json.dump({"doc_id": doc_id, "chat_history": chat_history}, f)

        return ChatResponse(response=final_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")

@app.post("/reset_session")
def reset_session():
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
            return {"message": "Session file cleared."}
        else:
            return {"message": "Session file was already empty."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")

@app.get("/")
def root():
    return {"message": "AskYourPDF + Groq API is running"}

@app.get("/test")
def test():
    return {
        "routes": [route.path for route in app.routes if isinstance(route, APIRoute)],
        "document_ids": document_ids
    }
