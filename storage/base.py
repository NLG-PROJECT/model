from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class StorageInterface(ABC):
    """Base interface for storage implementations."""
    
    @abstractmethod
    async def store_chunk(self, chunk_id: str, chunk: str, metadata: Dict[str, Any] = None, ttl: int = None) -> None:
        """Store a chunk in storage.
        
        Args:
            chunk_id: Unique chunk ID
            chunk: Text content of the chunk
            metadata: Optional metadata for the chunk
            ttl: Optional time-to-live in seconds
        """
        pass
    
    @abstractmethod
    async def store_chunk_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Store a chunk's embedding.
        
        Args:
            chunk_id: Unique chunk ID
            embedding: Vector embedding of the chunk
        """
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk and its metadata.
        
        Args:
            chunk_id: Unique chunk ID
            
        Returns:
            Dictionary containing chunk content and metadata, or None if not found
        """
        pass
    
    @abstractmethod
    async def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Retrieve a chunk's embedding.
        
        Args:
            chunk_id: Unique chunk ID
            
        Returns:
            Numpy array containing the embedding, or None if not found
        """
        pass
    
    @abstractmethod
    async def store_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Store document metadata."""
        pass
    
    @abstractmethod
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata."""
        pass
    
    @abstractmethod
    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated chunks."""
        pass
    
    @abstractmethod
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunk IDs
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the storage service is healthy."""
        pass 