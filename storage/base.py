from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class StorageInterface(ABC):
    """Base interface for storage implementations."""
    
    @abstractmethod
    def store_chunk(self, chunk_id: str, chunk_data: Dict[str, Any]) -> bool:
        """Store a single chunk of data."""
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single chunk of data."""
        pass
    
    @abstractmethod
    def store_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Store document metadata."""
        pass
    
    @abstractmethod
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata."""
        pass
    
    @abstractmethod
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs associated with a document."""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated chunks."""
        pass
    
    @abstractmethod
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query embedding."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the storage service is healthy."""
        pass 