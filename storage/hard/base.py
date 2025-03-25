from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, BinaryIO
import logging

logger = logging.getLogger(__name__)

class HardStorageInterface(ABC):
    """Base interface for hard storage implementations."""
    
    @abstractmethod
    def store_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Store a document and its metadata.
        
        Args:
            doc_id: Unique identifier for the document
            content: The document content
            metadata: Document metadata (title, author, etc.)
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document and its metadata.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Dictionary containing document content and metadata, or None if not found
        """
        pass
    
    @abstractmethod
    def store_chunk(self, chunk_id: str, content: str, doc_id: str) -> bool:
        """Store a document chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: The chunk content
            doc_id: ID of the parent document
        """
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Dictionary containing chunk content and metadata, or None if not found
        """
        pass
    
    @abstractmethod
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs associated with a document.
        
        Args:
            doc_id: ID of the parent document
            
        Returns:
            List of chunk IDs
        """
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated chunks.
        
        Args:
            doc_id: ID of the document to delete
        """
        pass
    
    @abstractmethod
    def store_file(self, file_id: str, file_content: BinaryIO, metadata: Dict[str, Any]) -> bool:
        """Store a file (e.g., PDF, DOCX).
        
        Args:
            file_id: Unique identifier for the file
            file_content: Binary file content
            metadata: File metadata (filename, mime_type, etc.)
        """
        pass
    
    @abstractmethod
    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a file and its metadata.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            Dictionary containing file content and metadata, or None if not found
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the storage service is healthy."""
        pass 