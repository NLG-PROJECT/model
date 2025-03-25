from typing import List, Dict, Any, Optional
from fastapi import UploadFile
import logging
from datetime import datetime
import uuid
import tempfile
from pathlib import Path
from storage.manager import StorageManager
from config.storage import StorageConfig, RedisConfig, GoogleDriveConfig
from processors.factory import ProcessorFactory
from utils.chunking import get_chunker
from embeddings.factory import EmbeddingServiceFactory
import os
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

class FileService:
    """Service for handling file operations."""
    
    def __init__(self, storage_manager: StorageManager):
        """Initialize file service.
        
        Args:
            storage_manager: Storage manager instance
        """
        self.storage_manager = storage_manager
        
        # Initialize chunker with configuration
        self.chunker = get_chunker(
            strategy="recursive",  # Can be made configurable if needed
            chunk_size=int(os.getenv("STORAGE_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("STORAGE_CHUNK_OVERLAP", "50")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100"))
        )
        
        # Initialize embedding service
        self.embedding_service = EmbeddingServiceFactory.get_service(
            name=os.getenv("EMBEDDING_PROVIDER", "ollama"),
            model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # Validate embedding service connection
        if not self.embedding_service.validate_connection():
            logger.warning("Failed to validate embedding service connection")
    
    async def process_files(self, files: List[UploadFile], save_to_drive: bool = True) -> List[Dict[str, Any]]:
        """Process and store multiple uploaded files.
        
        Args:
            files: List of uploaded files
            save_to_drive: Whether to save files to Google Drive
            
        Returns:
            List of processing results
        """
        results = []
        
        # Process files in parallel
        tasks = []
        for file in files:
            task = self._process_single_file(file, save_to_drive)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "filename": files[i].filename,
                    "doc_id": str(uuid.uuid4()),
                    "chunks_count": 0,
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_file(self, file: UploadFile, save_to_drive: bool) -> Dict[str, Any]:
        """Process a single file.
        
        Args:
            file: Uploaded file
            save_to_drive: Whether to save to Google Drive
            
        Returns:
            Processing result
        """
        try:
            # Read file content
            file_content = await file.read()
            
            # Get file extension safely
            file_extension = ""
            if file.filename:
                try:
                    # Handle PDF files specifically
                    if file.filename.lower().endswith('.pdf'):
                        file_extension = '.pdf'
                    else:
                        file_extension = Path(str(file.filename)).suffix
                except Exception:
                    # If we can't get the extension, try to determine from content type
                    content_type = file.content_type or ""
                    if "pdf" in content_type.lower():
                        file_extension = ".pdf"
                    elif "word" in content_type.lower():
                        file_extension = ".docx"
                    elif "excel" in content_type.lower():
                        file_extension = ".xlsx"
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                # Get appropriate processor
                processor = ProcessorFactory.get_processor(Path(temp_path))
                if not processor:
                    raise ValueError(f"No processor found for file type: {file.filename}")
                
                # Process document
                processed_data = await processor.process(temp_path)
                
                # Prepare metadata
                metadata = {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(file_content),
                    "created_at": datetime.utcnow().isoformat(),
                    "storage_type": "permanent" if save_to_drive else "temporary"
                }
                
                # Store document
                doc_id = await self.storage_manager.store_document(
                    file_content,
                    file.filename,
                    metadata,
                    save_to_drive=save_to_drive
                )
                
                # Process chunks if text was extracted
                chunks_count = 0
                if processed_data.get("text"):
                    chunks = self._create_chunks(processed_data["text"])
                    if chunks:
                        # Generate embeddings
                        embeddings = await self._generate_embeddings(chunks)
                        
                        # Store chunks and embeddings
                        chunk_ids = await self.storage_manager.store_chunks(
                            doc_id,
                            chunks,
                            embeddings,
                            [{"doc_id": doc_id} for _ in chunks]
                        )
                        chunks_count = len(chunk_ids)
                
                return {
                    "filename": file.filename,
                    "doc_id": doc_id,
                    "chunks_count": chunks_count,
                    "status": "success"
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return {
                "filename": file.filename,
                "doc_id": str(uuid.uuid4()),  # Generate a unique ID even for failed files
                "chunks_count": 0,  # Set to 0 for failed files
                "status": "error",
                "error": str(e)
            }
    
    def _create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create text chunks with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for chunk in chunks:
            embedding = await self._get_embedding(chunk)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Use your embedding service here
            # This is a placeholder
            return np.random.rand(1536)  # Example embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def get_file_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file information by document ID.
        
        Args:
            doc_id: Document ID (UUID)
            
        Returns:
            File information or None if not found
        """
        try:
            doc_data = await self.storage_manager.get_document(doc_id)
            if not doc_data:
                return None
                
            return {
                "doc_id": doc_id,
                "gdrive_file_id": doc_data["metadata"].get("gdrive_file_id", ""),
                "filename": doc_data["metadata"]["filename"],
                "content_type": doc_data["metadata"]["content_type"],
                "size": doc_data["metadata"]["size"],
                "created_at": datetime.fromisoformat(doc_data["metadata"]["created_at"]),
                "updated_at": datetime.utcnow(),
                "metadata": doc_data["metadata"]
            }
        except Exception as e:
            logger.error(f"Error retrieving file info for {doc_id}: {e}")
            return None
    
    async def list_files(self) -> List[Dict[str, Any]]:
        """List all stored files.
        
        Returns:
            List of file information
        """
        try:
            files = await self.storage_manager.list_documents()
            return [
                {
                    "doc_id": file["doc_id"],  # Main identifier at root level
                    "filename": file["metadata"]["filename"],
                    "content_type": file["metadata"]["content_type"],
                    "size": file["metadata"]["size"],
                    "created_at": file["metadata"]["created_at"],
                    "updated_at": datetime.utcnow().isoformat(),
                    "metadata": file["metadata"]  # Keep all metadata including gdrive_file_id
                }
                for file in files
            ]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def delete_file(self, doc_id: str) -> bool:
        """Delete a file by document ID.
        
        Args:
            doc_id: Document ID (UUID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.storage_manager.delete_document(doc_id)
        except Exception as e:
            logger.error(f"Error deleting file {doc_id}: {e}")
            return False
    
    async def get_file(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a single file by document ID.
        
        Args:
            doc_id: The document ID (UUID) to retrieve
            
        Returns:
            Dictionary containing file information and content, or None if not found
        """
        try:
            # Get document data using doc_id
            doc_data = await self.storage_manager.get_document(doc_id)
            if doc_data:
                metadata = doc_data.get("metadata", {})
                return {
                    "doc_id": doc_id,  # Main identifier at root level
                    "filename": metadata.get("filename", ""),
                    "content_type": metadata.get("content_type", ""),
                    "size": metadata.get("size", 0),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": datetime.utcnow().isoformat(),
                    "content": doc_data.get("content", ""),
                    "metadata": metadata  # Keep all metadata including gdrive_file_id
                }
            
            logger.warning(f"Document {doc_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving file {doc_id}: {e}")
            return None 