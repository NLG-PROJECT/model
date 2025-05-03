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
        request_body = [{"sender": "user", "message": ASK_YOUR_SUMMARY_PROMPT}]
        ask_response = requests.post(ask_url, headers=headers, json=request_body)

        if ask_response.status_code != 200:
            print(f"🔴 AskYourPDF response: {ask_response.text}")
            raise HTTPException(status_code=ask_response.status_code, detail="Failed to fetch insight sections from AskYourPDF")

        context = ask_response.json()["answer"]["message"]
        print(f"🔴 Context: {context}")
        # Step 2: Format context into structured prompt for Groq
        # groq_prompt = GROQ_MARKET_SUMMARY_PROMPT_TEMPLATE(context)

        # client = Groq(api_key=GROQ_CLOUD_API_KEY)
        # completion = client.chat.completions.create(
        #     model="llama3-70b-8192",
        #     messages=[
        #         {"role": "system", "content": "You are FinGroq, an elite financial analysis assistant."},
        #         {"role": "user", "content": groq_prompt},
        #     ],
        #     temperature=0.4,
        #     max_tokens=1800,
        #     top_p=0.9,
        #     stream=False
        # )

        # summary = completion.choices[0].message.content.strip()

        # Step 3: Save result into session file
        session_data["market_summary"] = context
        with open(SESSION_FILE, "w") as f:
            json.dump(session_data, f)

        return {"message": "Market summary generated successfully.", "summary": context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate market summary: {str(e)}")

@app.post("/generate/risk-factors")
def generate_risk_factors():
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
        request_body = [{"sender": "user", "message": ASK_RISK_FACTORS_SUMMARY_PROMPT}]
        ask_response = requests.post(ask_url, headers=headers, json=request_body)

        if ask_response.status_code != 200:
            print(f"🔴 AskYourPDF response: {ask_response.text}")
            raise HTTPException(status_code=ask_response.status_code, detail="Failed to fetch insight sections from AskYourPDF")

        context = ask_response.json()["answer"]["message"]
        print(f"🔴 Context: {context}")
        # Step 2: Format context into structured prompt for Groq
        # groq_prompt = GROQ_MARKET_SUMMARY_PROMPT_TEMPLATE(context)

        # client = Groq(api_key=GROQ_CLOUD_API_KEY)
        # completion = client.chat.completions.create(
        #     model="llama3-70b-8192",
        #     messages=[
        #         {"role": "system", "content": "You are FinGroq, an elite financial analysis assistant."},
        #         {"role": "user", "content": groq_prompt},
        #     ],
        #     temperature=0.4,
        #     max_tokens=1800,
        #     top_p=0.9,
        #     stream=False
        # )

        # summary = completion.choices[0].message.content.strip()

        # Step 3: Save result into session file
        session_data["risk_factors"] = context
        with open(SESSION_FILE, "w") as f:
            json.dump(session_data, f)

        return {"message": "Risk factors generated successfully.", "summary": context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate risk factors: {str(e)}")

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
    "You are assisting with extracting relevant sections from an SEC filing to support downstream market analysis.\n\n"
    "Your job is to extract only raw document content that aligns with the categories below. Prioritize paragraphs, tables, or bullet points that include numeric or factual detail (e.g., revenue figures, percentages, unit volumes, growth rates, dates, or dollar values).\n"
    "If no matching content exists for a category, skip it entirely.\n"
    "Do not paraphrase, summarize, or explain anything.\n\n"

    "Extract content relevant to:\n"
    "1. Industry Overview\n"
    "2. Competitive Landscape\n"
    "3. Customer Segments\n"
    "4. Revenue Breakdown\n"
    "5. Risks and Challenges\n"
    "6. Geographic Exposure\n"
    "7. Macroeconomic Sensitivity\n"
    "8. Innovation and R&D\n"
    "9. Strategic Initiatives\n"
    "10. ESG and Legal Disclosures\n\n"

    "Focus on detailed disclosures that can be used to summarize the company’s market position, growth dynamics, customer base, and external risks. Prefer content with numbers over qualitative statements.\n\n"

    "Example (for Competitive Landscape):\n"
    "Intel’s major competitors include Advanced Micro Devices, Inc. (AMD), NVIDIA Corporation, and companies using ARM architecture such as Apple Inc. and Qualcomm Incorporated. The company also competes with other semiconductor foundries, including Taiwan Semiconductor Manufacturing Company Limited (TSMC), Samsung Electronics Co., Ltd., and GlobalFoundries Inc.\n"
    "Intel’s market share in the data center segment declined in 2023 due to increased adoption of GPU-accelerated workloads and ARM-based server processors.\n\n"

    "Return only raw document excerpts. Do not insert any new text."
)




def GROQ_MARKET_SUMMARY_PROMPT_TEMPLATE(context: str) -> str:
    return f"""
<|identity|>
You are FinGroq, a world-class financial analyst AI used by hedge funds, auditors, and M&A advisors to produce structured, fluent, and numerically grounded market summaries from SEC filings.

<|objective|>
Your job is to produce a clean, narrative-style market analysis summary based entirely on the SEC filing content provided. Do not invent facts. Every statement must be backed by the document.

<|instructions|>
Generate a well-structured summary that includes:
1. Industry Overview
2. Competitive Landscape
3. Customer Segments
4. Revenue Breakdown
5. Risks and Challenges
6. Geographic Exposure
7. Macroeconomic Sensitivity
8. Innovation and R&D
9. Strategic Initiatives
10. ESG and Legal Disclosures (if available)

Guidelines:
- Use fluent, analyst-grade language that flows naturally.
- Include numeric data (revenue, percentages, share figures) wherever possible.
- Skip any section if no valid content exists—do not insert filler.
- Do not include interpretation, speculation, or external data.

<|formatting_guidelines|>
- Use markdown-style section headers (###) for each section.
- Write natural paragraphs (not just bullet points) while staying factual.
- Maintain clarity, objectivity, and conciseness.

<|one_shot_example|>
### Industry Overview
- Intel operates within the global semiconductor industry, producing products such as microprocessors, chipsets, and SoCs based on Intel architecture.
- The sector is undergoing transformation driven by ubiquitous compute, pervasive connectivity, AI, and edge infrastructure.
- Risks include high R&D costs, supply chain fragility, and geopolitical tensions like U.S.–China trade disputes.

### Competitive Landscape
- Intel’s major competitors include AMD, NVIDIA, Qualcomm, and TSMC.
- The company experienced market share erosion due to shifts toward GPUs and ARM-based solutions.

### Customer Segments
- 40% of 2023 net revenue came from three customers: Dell (19%), Lenovo (11%), and HP (10%).
- The company also disclosed revenue distribution across China ($14.9B), U.S. ($13.96B), and other key regions.

### Revenue Breakdown
- 2023 revenue was $54.2B, down 14% from $63.1B in 2022.
- Segment performance: CCG ($29.3B), DCAI ($15.5B), NEX ($5.8B), Mobileye ($2.1B), IFS ($952M).
- Revenue declines were driven by reduced demand in notebook and server segments.

### Risks and Challenges
- Intel faces risks from margin compression, competitive GPU adoption, and ecosystem shifts to ARM.
- R&D spending reached $16.0B in 2023 (29.6% of net revenue).
- Capital projects, like the Arizona Fab, carry long-term risk with $29B in unrecognized commitments.

### Geographic Exposure
- Revenue was concentrated in China and the U.S., with additional exposure in Singapore, Taiwan, and other markets.

### Macroeconomic Sensitivity
- Geopolitical tensions, inflation, and trade policy affect demand and manufacturing capacity.

### Innovation and R&D
- The company is heavily investing in foundry services and next-gen chip architectures.
- R&D intensity remains high to support turnaround strategy by 2025.

### Strategic Initiatives
- Intel is restructuring to separate its manufacturing and product design operations.
- It has announced multi-billion-dollar fabs in Arizona and Ohio.

<|context|>
<<document>>
{context}
<</document>>

<|output|>
Return your summary in the same format shown above.
"""

# ASK_YOUR_SUMMARY_PROMPT = """
# <|identity|>
# You are FinGroq, a world-class financial analyst AI used by hedge funds, auditors, and M&A advisors to produce structured, fluent, and numerically grounded market summaries from SEC filings.

# <|objective|>
# Your job is to generate a professional market analysis summary strictly based on SEC filing content. Every statement must be supported by the document—do not fabricate, speculate, or interpret beyond what is present.

# <|instructions|>
# Write a structured summary that includes the following sections:
# 1. Industry Overview
# 2. Competitive Landscape
# 3. Customer Segments
# 4. Revenue Breakdown
# 5. Risks and Challenges
# 6. Geographic Exposure
# 7. Macroeconomic Sensitivity
# 8. Innovation and R&D
# 9. Strategic Initiatives
# 10. ESG and Legal Disclosures (if applicable)

# You must follow the format, tone, and style shown in the example below. Each section should be fluent, fact-based, and anchored in numeric or structural detail from the filing. If the document lacks content for a section, omit that section entirely without adding placeholders or assumptions.

# <|formatting_guidelines|>
# - Use markdown-style section headers (###).
# - Write in fluent analyst-grade prose (not bullet points), using short paragraphs.
# - Include numeric data (e.g., revenue, % change, dollar values) wherever available.
# - Skip empty sections; do not generate filler or speculative text.
# - Model your output on the reference example below in both structure and depth.

# <|example|>
# ### Industry Overview
# Intel operates within the global semiconductor industry, producing products such as microprocessors, chipsets, and SoCs based on Intel architecture.  
# The semiconductor sector is undergoing structural transformation driven by five "superpowers": ubiquitous compute, pervasive connectivity, cloud-to-edge infrastructure, AI, and sensing.  
# Market growth is influenced by a generational shift to “system of chips,” increased customization for AI workloads, and vertically integrated designs by OEMs and CSPs.  
# Risks include high R&D costs, supply chain fragility, and geopolitical tensions (e.g., US-China trade relations).

# ### Competitive Landscape
# Major competitors: AMD, NVIDIA, Qualcomm, TSMC, Samsung, Global Foundries, UMC, SMIC.  
# Intel’s DCAI market share was impacted by a shift in spending toward GPUs and ARM-based solutions (e.g., Apple’s in-house chips).  
# Competitive pressure is high across performance, price, integration, ecosystem, and time-to-market. Intel is investing in regaining leadership by 2025.

# ### Customer Segments
# 40% of 2023 net revenue came from Dell (19%), Lenovo (11%), and HP (10%)—indicating high customer concentration.  
# Regional revenue breakdown: China ($14.9B), U.S. ($13.96B), Singapore ($8.6B), Taiwan ($6.9B), Others ($9.9B).

# ### Revenue Breakdown
# Total revenue in 2023 was $54.2B, down 14% from $63.1B in 2022.  
# Revenue by segment (2023):  
# - Client Computing Group (CCG): $29.3B  
# - Data Center and AI (DCAI): $15.5B  
# - Network and Edge (NEX): $5.8B  
# - Mobileye: $2.1B  
# - Intel Foundry Services (IFS): $952M  
# Declines attributed to lower demand in notebooks (-5% volume, -5% ASP), desktops (-9% volume), and server CPUs (-37% volume).

# ### Risks and Challenges
# Revenue and margin pressures from fierce competition, GPU adoption, and ecosystem shifts (e.g., ARM).  
# Operational risks include capital-intensive manufacturing, inventory write-downs, and limited foundry experience.  
# R&D expenses totaled $16.0B in 2023 (29.6% of net revenue), showing commitment to innovation but also financial burden.  
# Strategic projects (e.g., Arizona Fab) come with unrecognized commitments of up to $29.0B.

# ### Geographic Exposure
# Revenue was concentrated in China and the U.S., with additional exposure in Singapore, Taiwan, and other markets.

# ### Macroeconomic Sensitivity
# Geopolitical tensions, inflation, and trade policy affect demand and manufacturing capacity.

# ### Innovation and R&D
# The company is heavily investing in foundry services and next-gen chip architectures.  
# R&D intensity remains high to support turnaround strategy by 2025.

# ### Strategic Initiatives
# Intel is restructuring to separate its manufacturing and product design operations.  
# It has announced multi-billion-dollar fabs in Arizona and Ohio.

# ### ESG and Legal Disclosures
# (No relevant disclosures were available in the source content.)
# """

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
