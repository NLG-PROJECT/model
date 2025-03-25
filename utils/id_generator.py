import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class IDInfo:
    """Information about a generated ID."""
    id: str
    timestamp: datetime
    type: str  # 'doc' or 'chunk'
    parent_id: Optional[str] = None

class IDGenerator:
    """Generates unique IDs for documents and chunks."""
    
    @staticmethod
    def generate_doc_id() -> IDInfo:
        """Generate a unique document ID."""
        doc_id = f"doc_{uuid.uuid4().hex}"
        return IDInfo(
            id=doc_id,
            timestamp=datetime.now(),
            type="doc"
        )
    
    @staticmethod
    def generate_chunk_id(parent_doc_id: str, chunk_number: int) -> IDInfo:
        """Generate a unique chunk ID."""
        chunk_id = f"chunk_{parent_doc_id}_{chunk_number}_{uuid.uuid4().hex[:8]}"
        return IDInfo(
            id=chunk_id,
            timestamp=datetime.now(),
            type="chunk",
            parent_id=parent_doc_id
        )
    
    @staticmethod
    def generate_vector_id(chunk_id: str) -> str:
        """Generate a vector ID for a chunk."""
        return f"vector_{chunk_id}"
    
    @staticmethod
    def generate_table_id(doc_id: str, table_number: int) -> str:
        """Generate a table ID."""
        return f"table_{doc_id}_{table_number}"
    
    @staticmethod
    def generate_chart_id(doc_id: str, chart_number: int) -> str:
        """Generate a chart ID."""
        return f"chart_{doc_id}_{chart_number}"
    
    @staticmethod
    def parse_id(id_string: str) -> dict:
        """Parse an ID string to extract its components."""
        parts = id_string.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid ID format: {id_string}")
        
        id_type = parts[0]
        if id_type == "doc":
            return {
                "type": "doc",
                "uuid": parts[1]
            }
        elif id_type == "chunk":
            return {
                "type": "chunk",
                "doc_id": parts[1],
                "chunk_number": int(parts[2]),
                "uuid": parts[3]
            }
        elif id_type == "vector":
            return {
                "type": "vector",
                "chunk_id": "_".join(parts[1:])
            }
        elif id_type == "table":
            return {
                "type": "table",
                "doc_id": parts[1],
                "table_number": int(parts[2])
            }
        elif id_type == "chart":
            return {
                "type": "chart",
                "doc_id": parts[1],
                "chart_number": int(parts[2])
            }
        else:
            raise ValueError(f"Unknown ID type: {id_type}")
    
    @staticmethod
    def is_valid_id(id_string: str) -> bool:
        """Check if an ID string is valid."""
        try:
            IDGenerator.parse_id(id_string)
            return True
        except ValueError:
            return False 