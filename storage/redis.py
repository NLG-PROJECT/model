from typing import List, Dict, Any, Optional
import json
import redis
import numpy as np
from .base import StorageInterface
import logging
import os
from urllib.parse import urlparse
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from config.storage import RedisConfig

logger = logging.getLogger(__name__)

class RedisStorage(StorageInterface):
    """Redis implementation of the storage interface."""
    
    def __init__(self, config: RedisConfig):
        """Initialize Redis storage.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self.client = redis.Redis(
            host=config.host,
            port=config.port,
            password=config.password,
            db=config.db,
            decode_responses=False  # Need binary mode for vector storage
        )
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize Redis vector search index."""
        try:
            # Check if index exists
            self.client.ft(self.config.index_name).info()
            logger.info(f"Redis index '{self.config.index_name}' already exists.")
        except:
            # Create vector index
            schema = (
                TextField("chunk"),
                TextField("source"),
                TextField("doc_id"),
                VectorField("embedding", 
                          "HNSW", {
                              "TYPE": "FLOAT32", 
                              "DIM": self.config.embedding_dimension, 
                              "DISTANCE_METRIC": "COSINE"
                          })
            )
            
            definition = IndexDefinition(prefix=["chunk:"], index_type=IndexType.HASH)
            
            self.client.ft(self.config.index_name).create_index(
                fields=schema,
                definition=definition
            )
            logger.info(f"Created new Redis index '{self.config.index_name}'")
    
    def _get_chunk_key(self, chunk_id: str) -> str:
        """Get Redis key for a chunk."""
        return f"{self.config.prefix}:chunk:{chunk_id}"
    
    def _get_doc_metadata_key(self, doc_id: str) -> str:
        """Get Redis key for document metadata."""
        return f"{self.config.prefix}:doc:{doc_id}:metadata"
    
    def _get_doc_chunks_key(self, doc_id: str) -> str:
        """Get Redis key for document chunks list."""
        return f"{self.config.prefix}:doc:{doc_id}:chunks"
    
    def store_chunk(self, chunk_id: str, chunk_data: Dict[str, Any]) -> bool:
        """Store a single chunk of data."""
        try:
            key = self._get_chunk_key(chunk_id)
            # Convert embedding to string for storage
            if "embedding" in chunk_data:
                chunk_data["embedding"] = chunk_data["embedding"].tolist()
            self.client.set(key, json.dumps(chunk_data))
            return True
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
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
        """Store document metadata in Redis.
        
        Args:
            doc_id: Document ID
            metadata: Document metadata
            
        Returns:
            True if successful
        """
        try:
            key = f"{self.config.prefix}:doc:{doc_id}:metadata"
            self.client.set(key, json.dumps(metadata))
            logger.info(f"Stored metadata for document {doc_id}: {metadata}")
            return True
        except Exception as e:
            logger.error(f"Error storing document metadata: {e}")
            return False
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata from Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        try:
            key = self._get_doc_metadata_key(doc_id)
            data = self.client.get(key)
            if not data:
                return None
            
            return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {e}")
            return None
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs associated with a document."""
        try:
            key = self._get_doc_chunks_key(doc_id)
            chunks = self.client.smembers(key)
            return [chunk.decode() for chunk in chunks]
        except Exception as e:
            logger.error(f"Error getting chunks for doc {doc_id}: {e}")
            return []
    
    async def store_chunk_embeddings(self, doc_id: str, chunk_ids: List[str], 
                                   embeddings: List[np.ndarray]) -> bool:
        """Store chunk embeddings in Redis.
        
        Args:
            doc_id: Document ID
            chunk_ids: List of chunk IDs
            embeddings: List of chunk embeddings
            
        Returns:
            True if successful
        """
        try:
            pipeline = self.client.pipeline()
            
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                key = f"chunk:{chunk_id}"
                pipeline.hset(key, mapping={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "embedding": embedding.tobytes()
                })
            
            pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Error storing chunk embeddings: {e}")
            return False
    
    async def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve chunks from Redis cache.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks with their embeddings
        """
        try:
            # Search for chunks belonging to this document
            query = f"@doc_id:{doc_id}"
            results = self.client.ft(self.config.index_name).search(
                Query(query).return_fields("chunk", "embedding", "chunk_id")
            )
            
            chunks = []
            for doc in results.docs:
                chunks.append({
                    "chunk_id": doc.chunk_id,
                    "content": doc.chunk,
                    "embedding": np.frombuffer(doc.embedding, dtype=np.float32)
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks from cache: {e}")
            return []
    
    async def cache_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Cache chunks in Redis.
        
        Args:
            doc_id: Document ID
            chunks: List of chunks to cache
            
        Returns:
            True if successful
        """
        try:
            pipeline = self.client.pipeline()
            
            for chunk in chunks:
                key = f"chunk:{chunk['chunk_id']}"
                pipeline.hset(key, mapping={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "content": chunk["content"],
                    "embedding": chunk["embedding"].tobytes()
                })
                pipeline.expire(key, self.config.chunk_ttl)
            
            pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Error caching chunks: {e}")
            return False
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunk IDs
        """
        try:
            # Prepare the vector search query
            vector_query = (
                f"*=>[KNN {top_k} @embedding $vector AS score]"
            )
            
            # Execute the vector search
            query = (
                Query(vector_query)
                .dialect(2)  # Use Query dialect 2 for vector search
                .sort_by("score")
                .paging(0, top_k)
                .return_fields("chunk_id")
            )
            
            # Get query embedding (you'll need to implement this)
            query_embedding = self._get_query_embedding(query)
            params_dict = {"vector": query_embedding.tobytes()}
            
            # Execute search
            results = self.client.ft(self.config.index_name).search(query, params_dict)
            
            return [doc.chunk_id for doc in results.docs]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document data from Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            # Delete document metadata
            self.client.delete(f"doc:{doc_id}")
            
            # Delete chunks
            query = f"@doc_id:{doc_id}"
            results = self.client.ft(self.config.index_name).search(
                Query(query).return_fields("chunk_id")
            )
            
            pipeline = self.client.pipeline()
            for doc in results.docs:
                pipeline.delete(f"chunk:{doc.chunk_id}")
            
            pipeline.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Redis connection is healthy.
        
        Returns:
            True if healthy
        """
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding
        """
        # TODO: Implement query embedding generation
        # This should use the same embedding model as the chunks
        raise NotImplementedError("Query embedding generation not implemented")
    
    async def list_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """List metadata for all documents.
        
        Returns:
            Dictionary mapping document IDs to their metadata
        """
        try:
            # Get all document metadata keys
            pattern = f"{self.config.prefix}:doc:*:metadata"
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
                        doc_id = key.decode().split(":")[2]  # Extract doc_id from key
                        doc_metadata[doc_id] = json.loads(data)
                        logger.info(f"Retrieved metadata for document {doc_id}: {doc_metadata[doc_id]}")
                    else:
                        logger.warning(f"No data found for key {key}")
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
            
            logger.info(f"Successfully retrieved metadata for {len(doc_metadata)} documents")
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error listing document metadata: {e}")
            return {} 