import logging
from typing import Dict, Any, List, Optional, BinaryIO
import uuid
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from .hard import HardStorageFactory
from .redis import LocalRedisStorage
from config.storage import StorageConfig

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages document storage across Redis and Google Drive."""
    
    def __init__(self, config: StorageConfig):
        """Initialize storage manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.hard_storage = HardStorageFactory.create_storage(
            'gdrive',
            credentials_path=config.gdrive.credentials_path,
            root_folder_id=config.gdrive.root_folder_id
        )
        self.redis_storage = LocalRedisStorage(config.redis)
        
        # Ensure base path exists
        self.config.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_document(self, file_content: BinaryIO, filename: str, metadata: Dict[str, Any], save_to_drive: bool = True) -> str:
        """Store a document and its metadata.
        
        Args:
            file_content: The document file content (BinaryIO or bytes)
            filename: Original filename
            metadata: Document metadata
            save_to_drive: Whether to save to Google Drive
            
        Returns:
            Document ID (UUID)
        """
        doc_id = str(uuid.uuid4())
        
        try:
            # Get content as bytes
            if hasattr(file_content, 'read'):
                content = file_content.read()
            else:
                content = file_content
            
            # Store file in Google Drive if requested
            gdrive_file_id = None
            if save_to_drive:
                gdrive_file_id = self.hard_storage.store_document(content, filename, metadata)
                if not gdrive_file_id:
                    raise Exception("Failed to store file in Google Drive")
            
            # Prepare metadata for Redis
            redis_metadata = {
                **metadata,  # Spread original metadata first
                "filename": filename,
                "upload_date": datetime.utcnow().isoformat(),
                "gdrive_file_id": gdrive_file_id,  # Will be None if save_to_drive is False
                "content_type": metadata.get("content_type", ""),
                "size": metadata.get("size", 0),
                "created_at": metadata.get("created_at", datetime.utcnow().isoformat()),
                "storage_type": "permanent" if save_to_drive else "temporary"
            }
            
            # Store document metadata in Redis
            await self.redis_storage.store_document_metadata(doc_id, redis_metadata)
            
            logger.info(f"Successfully stored document {doc_id} with storage type: {'permanent' if save_to_drive else 'temporary'}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document {doc_id}: {e}")
            raise
    
    async def store_chunks(self, doc_id: str, chunks: List[str], embeddings: List[np.ndarray], chunk_metadata: List[Dict[str, Any]] = None) -> List[str]:
        """Store document chunks and their embeddings.
        
        Args:
            doc_id: Document ID
            chunks: List of text chunks
            embeddings: List of chunk embeddings
            chunk_metadata: List of metadata for each chunk
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        
        try:
            # Store chunks in Google Drive
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                metadata = chunk_metadata[i] if chunk_metadata and i < len(chunk_metadata) else {}
                metadata.update({
                    "doc_id": doc_id,
                    "position": i,
                    "total_chunks": len(chunks)
                })
                
                # Store chunk in Google Drive (synchronous)
                stored_chunk_id = self.hard_storage.store_chunk(chunk_id, chunk, doc_id, metadata)
                if not stored_chunk_id:
                    raise Exception(f"Failed to store chunk {i} in Google Drive")
                
                chunk_ids.append(stored_chunk_id)
            
            # Store embeddings in Redis
            await self.redis_storage.store_chunk_embeddings(doc_id, chunk_ids, embeddings)
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks for document {doc_id}: {e}")
            raise
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document and its metadata.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data and metadata
        """
        try:
            # Try to get from cache first
            cache_key = f"doc:{doc_id}"
            cached_data = await self.redis_storage.get_chunk(cache_key)
            if cached_data:
                logger.debug(f"Retrieved document {doc_id} from cache")
                return cached_data
            
            # Get document metadata from Redis
            metadata = await self.redis_storage.get_document_metadata(doc_id)
            if not metadata:
                logger.warning(f"No metadata found for document {doc_id}")
                return None
            
            # Get Google Drive file ID from metadata
            gdrive_file_id = metadata.get("gdrive_file_id")
            if not gdrive_file_id:
                logger.warning(f"No Google Drive file ID found in metadata for {doc_id}")
                return None
            
            # Get document content from Google Drive (synchronous)
            doc_data = self.hard_storage.get_document(gdrive_file_id)
            if not doc_data:
                logger.warning(f"No content found for document {doc_id}")
                return None
            
            # Combine document data and metadata
            result = {
                "doc_id": doc_id,  # Ensure doc_id is at the root level
                "filename": metadata.get("filename", ""),
                "content_type": metadata.get("content_type", ""),
                "size": metadata.get("size", 0),
                "created_at": metadata.get("created_at", ""),
                "updated_at": datetime.utcnow().isoformat(),
                "content": doc_data.get("content", ""),
                "metadata": metadata  # Keep all metadata including gdrive_file_id
            }
            
            # Cache the result with a reasonable TTL (e.g., 1 hour)
            await self.redis_storage.store_chunk(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    async def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve document chunks and their embeddings.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks with their embeddings
        """
        try:
            # Try to get chunks from Redis cache first
            chunks = await self.redis_storage.get_chunks(doc_id)
            if chunks:
                return chunks
            
            # If not in cache, get from Google Drive
            chunk_ids = self.hard_storage.get_document_chunks(doc_id)
            chunks = []
            
            for chunk_id in chunk_ids:
                chunk_data = self.hard_storage.get_chunk(chunk_id)
                if chunk_data:
                    chunks.append(chunk_data)
            
            # Cache chunks in Redis if enabled
            if self.config.cache_enabled:
                await self.redis_storage.cache_chunks(doc_id, chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {doc_id}: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated data.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Google Drive
            success = self.hard_storage.delete_document(doc_id)
            if not success:
                return False
            
            # Delete from Redis
            await self.redis_storage.delete_document(doc_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with their metadata
        """
        try:
            # Get similar chunk IDs from Redis
            chunk_ids = await self.redis_storage.search_similar_chunks(query, top_k)
            
            # Get full chunk data from Google Drive
            chunks = []
            for chunk_id in chunk_ids:
                chunk_data = self.hard_storage.get_chunk(chunk_id)
                if chunk_data:
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if storage services are healthy.
        
        Returns:
            True if all services are healthy, False otherwise
        """
        try:
            # Check Google Drive
            if not self.hard_storage.health_check():
                return False
            
            # Check Redis
            if not await self.redis_storage.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents.
        
        Returns:
            List of documents with their metadata
        """
        try:
            logger.info("Starting to list documents")
            
            # Get all document metadata from Redis
            doc_metadata = await self.redis_storage.list_document_metadata()
            logger.info(f"Retrieved metadata for {len(doc_metadata)} documents from Redis")
            
            if not doc_metadata:
                logger.warning("No document metadata found")
                return []
            
            # Create document list without fetching content
            documents = []
            for doc_id, metadata in doc_metadata.items():
                try:
                    # Get Google Drive file ID from metadata
                    gdrive_file_id = metadata.get("gdrive_file_id")
                    if not gdrive_file_id:
                        logger.warning(f"No Google Drive file ID found in metadata for {doc_id}")
                        continue
                    
                    documents.append({
                        "doc_id": doc_id,  # Main identifier at root level
                        "filename": metadata.get("filename", ""),
                        "content_type": metadata.get("content_type", ""),
                        "size": metadata.get("size", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": datetime.utcnow().isoformat(),
                        "metadata": metadata  # Keep all metadata including gdrive_file_id
                    })
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
            
            logger.info(f"Successfully listed {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return [] 