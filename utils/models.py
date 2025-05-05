from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Model for chat request."""
    message: str
    option: str
    class Config:
        extra = "forbid"

class ChatResponse(BaseModel):
    """Model for chat response."""
    response: str

class FactCheckRequest(BaseModel):
    """Model for fact check request."""
    statement: str