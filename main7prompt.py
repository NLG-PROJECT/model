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

@app.post("/generate/market-summary")
def generate_market_summary():
    if not os.path.exists(SESSION_FILE):
        raise HTTPException(status_code=400, detail="No active session. Upload a document first.")

    try:
        with open(SESSION_FILE, "r") as f:
            session_data = json.load(f)
            doc_id = session_data.get("doc_id")

        if not doc_id:
            raise HTTPException(status_code=400, detail="Document ID missing in session.")

        # Step 1: Ask AskYourPDF for raw insight sections
        headers = {"x-api-key": ASKYOURPDF_API_KEY, "Content-Type": "application/json"}
        ask_url = f"{ASKYOURPDF_BASE_URL}/chat/{doc_id}?stream=False"
        request_body = [{"sender": "user", "message": ASK_MARKET_INSIGHT_SECTION_PROMPT}]
        ask_response = requests.post(ask_url, headers=headers, json=request_body)

        if ask_response.status_code != 200:
            raise HTTPException(status_code=ask_response.status_code, detail="Failed to fetch insight sections from AskYourPDF")

        context = ask_response.json()["answer"]["message"]

        # Step 2: Format context into structured prompt for Groq
        groq_prompt = GROQ_MARKET_SUMMARY_PROMPT_TEMPLATE(context)

        client = Groq(api_key=GROQ_CLOUD_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are FinGroq, an elite financial analysis assistant."},
                {"role": "user", "content": groq_prompt},
            ],
            temperature=0.4,
            max_tokens=1800,
            top_p=0.9,
            stream=False
        )

        summary = completion.choices[0].message.content.strip()

        # Step 3: Save result into session file
        session_data["market_summary"] = summary
        with open(SESSION_FILE, "w") as f:
            json.dump(session_data, f)

        return {"message": "Market summary generated successfully.", "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate market summary: {str(e)}")

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




## TEMPLATES
ASK_MARKET_INSIGHT_SECTION_PROMPT = (
    "You are assisting in extracting relevant sections from an SEC filing for downstream financial analysis.\n\n"
    "Your task is to ONLY return raw document content that is directly relevant to the following market analysis categories.\n"
    "Focus on paragraphs, tables, or bullet points that include quantitative or numeric details such as dollar amounts, percentages, time-based trends, or market comparisons.\n"
    "Do not summarize, interpret, or paraphrase.\n"
    "Do not include any explanation or formatting.\n"
    "If no relevant content is found for a section, skip that section entirely — do not include placeholders or labels.\n\n"
    "Extract and return only raw document content related to:\n\n"

    "1. Industry Overview\n"
    "   - Market size, growth rate, trends, and total addressable market (TAM)\n"
    "   - Regulatory changes with quantified or economic impact\n\n"

    "2. Competitive Landscape\n"
    "   - Names of competitors, market share figures, or any comparative analysis with numerical backing\n"
    "   - Statements about scale, pricing, product differentiation supported by data\n\n"

    "3. Customer Segments\n"
    "   - Types of customers, geographic or sector-based segmentation\n"
    "   - Revenue breakdown by customer type or concentration risks (with percentages or dollar values)\n\n"

    "4. Revenue Breakdown\n"
    "   - Revenue by product line, service category, or region\n"
    "   - Year-over-year changes, growth drivers, or decline factors with figures\n\n"

    "5. Risks and Challenges\n"
    "   - Regulatory, supply chain, operational, or market risks with financial implications\n"
    "   - Legal contingencies, impairment risks, or macro threats expressed quantitatively\n\n"

    "6. Geographic Exposure\n"
    "   - Revenue or cost distribution by country or region\n"
    "   - Quantified international performance or exposure disclosures\n\n"

    "7. Macroeconomic Sensitivity\n"
    "   - Inflation, interest rate, foreign exchange (FX), or economic cycle impacts with data\n\n"

    "8. Innovation and R&D Focus\n"
    "   - R&D spending levels, growth rates, and product pipeline statistics\n"
    "   - Number of patents, technologies, or investments in innovation\n\n"

    "9. Strategic Initiatives\n"
    "   - M&A transactions, capital investments, restructuring efforts with dollar amounts\n"
    "   - Strategic goals or performance targets with measurable values\n\n"

    "10. ESG and Legal Disclosures\n"
    "   - ESG metrics (e.g., emissions, diversity), regulatory compliance costs, or litigation risks with figures\n"
    "   - Fines, provisions, or audit outcomes backed by disclosed amounts\n\n"

    "Return only the raw extracted text that contains relevant information.\n"
    "Skip any section entirely if no relevant or numeric content exists.\n"
    "Prioritize data-rich and verbatim content over general commentary."
)


def GROQ_MARKET_SUMMARY_PROMPT_TEMPLATE(context: str) -> str:
    return f"""
<|identity|>
You are FinGroq, a world-class financial analyst AI used by hedge funds, auditors, and M&A advisory teams to generate precise and fluent market summaries from SEC filings.

<|objective|>
Your job is to produce a structured and readable market analysis summary strictly based on the content provided in the document context. Do not speculate or add any information that is not explicitly found in the input.

<|context|>
<<document>>
{context}
<</document>>

<|instructions|>
Based on the document above, write a **tagged market analysis summary** that clearly and fluently covers the following sections:

1. **Industry Overview**  
   - Describe the sector the company operates in, recent trends, and estimated market size  
   - Include relevant regulatory influences if discussed

2. **Competitive Landscape**  
   - Highlight major competitors or market players and any reported market share details  
   - Mention comparative advantages or disadvantages

3. **Customer Segments**  
   - Outline key customer groups and any mention of concentration risk or geographic segmentation  
   - Mention if the company relies heavily on any one segment or client type

4. **Revenue Breakdown**  
   - Summarize the company’s main revenue streams and any notable growth or decline  
   - Provide numerical figures when present (e.g., revenue per segment or YOY growth)

5. **Risks and Challenges**  
   - Present strategic, regulatory, or operational risks as discussed  
   - Focus on risks from technological disruption, supply chain, economic uncertainty, etc.

6. **Geographic Exposure**  
   - Explain which regions the company is most active in and any geopolitical or regional risks noted

7. **Macroeconomic Sensitivity**  
   - Identify the company’s dependence on broader economic factors such as inflation, interest rates, or economic cycles

8. **Innovation and R&D Focus**  
   - Summarize any investment in R&D, proprietary technologies, or product pipelines discussed

9. **Strategic Initiatives**  
   - Mention recent M&A activity, restructuring, partnerships, or strategic goals if disclosed

10. **ESG and Legal Disclosures**  
   - Present key environmental, social, or governance disclosures, including litigation or compliance risks

<|formatting_guidelines|>
- Each section should be **narrative-driven**, not just bullet points.
- Use **numeric data** from the document (e.g., $4.5B revenue, 23% growth) to back up claims.
- Do **not** hallucinate or create any numbers or statements not in the context.
- Avoid preambles, fluff, or repetition — stay crisp and data-grounded.
- Do **not** summarize the entire document — only what's relevant to the specified sections.
- Maintain a tone that reads fluently and professionally, like a human analyst would write in a research note.

<|language_tone|>
Use objective, precise, and neutral language. Each paragraph should flow naturally but remain densely informative. Avoid adjectives unless used in the document itself. Do not insert personal opinions or soft qualifiers.

<|output|>
Return the structured summary in this exact format:

<|market_summary|>

### Industry Overview
[short fluent paragraph or two, backed with figures]

### Competitive Landscape
[short fluent paragraph or two, with names, percentages, or factual comparisons]

### Customer Segments
[...]

### Revenue Breakdown
[...]

### Risks and Challenges
[...]

### Geographic Exposure
[...]

### Macroeconomic Sensitivity
[...]

### Innovation and R&D Focus
[...]

### Strategic Initiatives
[...]

### ESG and Legal Disclosures
[...]

<|/market_summary|>
"""
