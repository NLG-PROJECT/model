from typing import List, Dict, Any, Optional
import json
import numpy as np
from ..base import StorageInterface
import logging

logger = logging.getLogger(__name__)

class BaseRedisStorage(StorageInterface):
    """Base class for Redis storage implementations."""
    
    def __init__(self, prefix: str = "docstore"):
        """Initialize base Redis storage.
        
        Args:
            prefix: Prefix for all Redis keys.
        """
        self.prefix = prefix
        self.client = None  # Will be set by subclasses
    
    def _get_chunk_key(self, chunk_id: str) -> str:
        """Get Redis key for a chunk."""
        return f"{self.prefix}:chunk:{chunk_id}"
    
    def _get_doc_metadata_key(self, doc_id: str) -> str:
        """Get Redis key for document metadata."""
        return f"{self.prefix}:doc:{doc_id}:metadata"
    
    def _get_doc_chunks_key(self, doc_id: str) -> str:
        """Get Redis key for document chunks list."""
        return f"{self.prefix}:doc:{doc_id}:chunks"
    
    async def store_chunk(self, chunk_id: str, chunk: str, metadata: Dict[str, Any] = None, ttl: int = None) -> None:
        """Store a chunk in Redis.
        
        Args:
            chunk_id: Unique chunk ID
            chunk: Text content of the chunk
            metadata: Optional metadata for the chunk
            ttl: Optional time-to-live in seconds
        """
        try:
            chunk_data = {
                "content": chunk,
                "metadata": metadata or {}
            }
            
            key = self._get_chunk_key(chunk_id)
            self.client.set(key, json.dumps(chunk_data))
            
            if ttl:
                self.client.expire(key, ttl)
            
            # Add to document's chunk list if doc_id is in metadata
            if metadata and "doc_id" in metadata:
                self.client.sadd(self._get_doc_chunks_key(metadata["doc_id"]), chunk_id)
            
            logger.debug(f"Stored chunk {chunk_id} in Redis")
            
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id} in Redis: {e}")
            raise
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single chunk of data."""
        try:
            key = self._get_chunk_key(chunk_id)
            data = self.client.get(key)
            if data:
                chunk_data = json.loads(data)
                # Convert embedding back to numpy array
                if "embedding" in chunk_data:
                    chunk_data["embedding"] = np.array(chunk_data["embedding"])
                return chunk_data
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    async def store_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Store document metadata."""
        try:
            key = self._get_doc_metadata_key(doc_id)
            self.client.set(key, json.dumps(metadata))
            return True
        except Exception as e:
            logger.error(f"Error storing metadata for doc {doc_id}: {e}")
            return False
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata."""
        try:
            key = self._get_doc_metadata_key(doc_id)
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error retrieving metadata for doc {doc_id}: {e}")
            return None
    
    async def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs associated with a document."""
        try:
            key = self._get_doc_chunks_key(doc_id)
            chunks = self.client.smembers(key)
            return [chunk.decode() for chunk in chunks]
        except Exception as e:
            logger.error(f"Error getting chunks for doc {doc_id}: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated chunks."""
        try:
            # Get all chunk IDs
            chunk_ids = await self.get_document_chunks(doc_id)
            
            # Delete chunks
            for chunk_id in chunk_ids:
                self.client.delete(self._get_chunk_key(chunk_id))
            
            # Delete document metadata and chunks list
            self.client.delete(self._get_doc_metadata_key(doc_id))
            self.client.delete(self._get_doc_chunks_key(doc_id))
            
            return True
        except Exception as e:
            logger.error(f"Error deleting doc {doc_id}: {e}")
            return False
    
    async def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query embedding."""
        try:
            # Get all chunk keys
            chunk_keys = self.client.keys(f"{self.prefix}:chunk:*")
            results = []
            
            for key in chunk_keys:
                data = self.client.get(key)
                if data:
                    chunk_data = json.loads(data)
                    if "embedding" in chunk_data:
                        chunk_embedding = np.array(chunk_data["embedding"])
                        similarity = np.dot(query_embedding, chunk_embedding)
                        results.append((similarity, chunk_data))
            
            # Sort by similarity and return top results
            results.sort(reverse=True)
            return [data for _, data in results[:limit]]
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if the storage service is healthy."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def list_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """List metadata for all documents.
        
        Returns:
            Dictionary mapping document IDs to their metadata
        """
        try:
            # Get all document metadata keys
            pattern = f"{self.prefix}:doc:*:metadata"
            logger.info(f"Searching for document metadata with pattern: {pattern}")
            
            metadata_keys = self.client.keys(pattern)
            logger.info(f"Found {len(metadata_keys)} metadata keys")
            
            if not metadata_keys:
                logger.warning("No document metadata found in Redis")
                return {}
            
            # Get metadata for each document
            doc_metadata = {}
            for key in metadata_keys:
                try:
                    data = self.client.get(key)
                    if data:
                        # Extract doc_id from key (format: prefix:doc:doc_id:metadata)
                        key_parts = key.decode().split(":")
                        if len(key_parts) >= 3:
                            doc_id = key_parts[2]
                            doc_metadata[doc_id] = json.loads(data)
                            logger.info(f"Retrieved metadata for document {doc_id}")
                    else:
                        logger.warning(f"No data found for key {key}")
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
            
            logger.info(f"Successfully retrieved metadata for {len(doc_metadata)} documents")
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error listing document metadata: {e}")
            return {}
    
    async def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks associated with a document."""
        try:
            chunk_ids = await self.get_document_chunks(doc_id)
            chunks = []
            for chunk_id in chunk_ids:
                chunk_data = await self.get_chunk(chunk_id)
                if chunk_data:
                    chunks.append(chunk_data)
            return chunks
        except Exception as e:
            logger.error(f"Error getting chunks for doc {doc_id}: {e}")
            return []
    
    async def store_chunk_embeddings(self, doc_id: str, chunk_ids: List[str], embeddings: List[np.ndarray]) -> bool:
        """Store chunk embeddings."""
        try:
            # Store chunk IDs in document's chunk list
            key = self._get_doc_chunks_key(doc_id)
            for chunk_id in chunk_ids:
                self.client.sadd(key, chunk_id)
            
            # Store embeddings for each chunk
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                chunk_key = self._get_chunk_key(chunk_id)
                chunk_data = {
                    "doc_id": doc_id,
                    "embedding": embedding.tolist()
                }
                self.client.set(chunk_key, json.dumps(chunk_data))
            
            return True
        except Exception as e:
            logger.error(f"Error storing chunk embeddings for doc {doc_id}: {e}")
            return False
    
    async def cache_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Cache document chunks in Redis."""
        try:
            # Store chunks in Redis
            for chunk in chunks:
                chunk_id = chunk.get("id")
                if chunk_id:
                    key = self._get_chunk_key(chunk_id)
                    self.client.set(key, json.dumps(chunk))
            
            return True
        except Exception as e:
            logger.error(f"Error caching chunks for doc {doc_id}: {e}")
            return False
    
    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by doc_id.
        
        Args:
            doc_id: Document ID to search for
            
        Returns:
            Document metadata if found, None otherwise
        """
        try:
            # Get all document metadata keys
            pattern = f"{self.prefix}:doc:*:metadata"
            metadata_keys = self.client.keys(pattern)
            
            if not metadata_keys:
                return None
            
            # Search for document with matching doc_id
            for key in metadata_keys:
                data = self.client.get(key)
                if data:
                    metadata = json.loads(data)
                    if metadata.get("doc_id") == doc_id:
                        return metadata
            
            return None
        except Exception as e:
            logger.error(f"Error getting document by doc_id {doc_id}: {e}")
            return None
    
    async def store_chunk_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Store a chunk's embedding in Redis.
        
        Args:
            chunk_id: Unique chunk ID
            embedding: Vector embedding of the chunk
        """
        try:
            key = self._get_chunk_key(chunk_id)
            chunk_data = {
                "embedding": embedding.tolist()
            }
            self.client.set(key, json.dumps(chunk_data))
            logger.debug(f"Stored embedding for chunk {chunk_id} in Redis")
        except Exception as e:
            logger.error(f"Error storing embedding for chunk {chunk_id} in Redis: {e}")
            raise
    
    async def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Retrieve a chunk's embedding from Redis.
        
        Args:
            chunk_id: Unique chunk ID
            
        Returns:
            Numpy array containing the embedding, or None if not found
        """
        try:
            key = self._get_chunk_key(chunk_id)
            data = self.client.get(key)
            if data:
                chunk_data = json.loads(data)
                if "embedding" in chunk_data:
                    return np.array(chunk_data["embedding"])
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for chunk {chunk_id} from Redis: {e}")
            return None 