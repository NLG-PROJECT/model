from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class FileMetadata(BaseModel):
    """File metadata schema."""
    filename: str
    content_type: str
    size: int
    created_at: str
    updated_at: str
    storage_type: str  # "permanent" or "temporary"
    gdrive_file_id: Optional[str] = None
    metadata: Dict[str, Any]

class FileResponse(BaseModel):
    """File response schema."""
    doc_id: str
    filename: str
    content_type: str
    size: int
    created_at: str
    updated_at: str
    metadata: FileMetadata

class ProcessedFile(BaseModel):
    """Processed file information."""
    filename: str
    doc_id: str
    chunks_count: int
    status: str
    error: Optional[str] = None

class UploadResponse(BaseModel):
    """Upload response schema."""
    message: str
    processed_files: List[ProcessedFile] 