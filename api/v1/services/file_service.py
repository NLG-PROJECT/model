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
import time

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
        """Process and store multiple uploaded files with optimized performance."""
        start_time = time.time()
        results = []
        
        try:
            # Process files in parallel
            tasks = []
            for file in files:
                task = self._process_single_file(file, save_to_drive)
                tasks.append(task)
            
            # Wait for all files to be processed
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing file: {result}")
                    processed_results.append({
                        "filename": "unknown",
                        "doc_id": str(uuid.uuid4()),
                        "chunks_count": 0,
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            # Log performance metrics
            total_time = time.time() - start_time
            success_count = sum(1 for r in processed_results if r["status"] == "success")
            error_count = len(processed_results) - success_count
            
            logger.info(f"File processing complete:")
            logger.info(f"- Total time: {total_time:.2f}s")
            logger.info(f"- Files processed: {len(processed_results)}")
            logger.info(f"- Success rate: {(success_count/len(processed_results))*100:.2f}%")
            logger.info(f"- Average time per file: {total_time/len(processed_results):.2f}s")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch file processing: {e}")
            raise
    
    async def _process_single_file(self, file: UploadFile, save_to_drive: bool = True) -> Dict[str, Any]:
        """Process a single file with optimized performance."""
        start_time = time.time()
        temp_path = None
        
        try:
            # Create temporary file with proper extension
            file_extension = Path(file.filename).suffix.lower()
            if not file_extension:
                # Try to determine from content type
                content_type = file.content_type or ""
                if "pdf" in content_type.lower():
                    file_extension = ".pdf"
                elif "word" in content_type.lower():
                    file_extension = ".docx"
                elif "excel" in content_type.lower():
                    file_extension = ".xlsx"
            
            logger.info(f"Processing file with extension: {file_extension}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Get processor for file type
            processor = ProcessorFactory.get_processor(Path(temp_path))
            if not processor:
                raise ValueError(f"No processor available for file type: {file_extension}")
            
            # Process document
            processed_data = await processor.process(temp_path)
            
            # Extract text and create chunks
            text = processed_data.get("content", {}).get("text", "")
            if not text:
                raise ValueError("No text extracted from document")
            
            chunks = self._create_chunks(text)
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # Generate embeddings in batches
            embeddings = await self._generate_embeddings(chunks)
            
            # Store document and chunks
            doc_id = await self.storage_manager.store_document(
                file_content=content,
                filename=file.filename,
                metadata={
                    "content_type": file.content_type,
                    "size": len(content),
                    "chunks_count": len(chunks),
                    "processing_time": time.time() - start_time,
                    "metadata": processed_data.get("metadata", {})
                },
                save_to_drive=save_to_drive
            )
            
            # Store chunks and embeddings
            await self.storage_manager.store_chunks(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings
            )
            
            return {
                "filename": file.filename,
                "doc_id": doc_id,
                "chunks_count": len(chunks),
                "status": "success",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return {
                "filename": file.filename,
                "doc_id": str(uuid.uuid4()),
                "chunks_count": 0,
                "status": "error",
                "error": str(e)
            }
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _create_chunks(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Create text chunks with smart boundaries and semantic preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (increased for better context)
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        logger.info(f"Starting smart chunking process with text length: {len(text)}")
        
        # Split text into semantic units (paragraphs, sections)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Define section markers (can be expanded based on document type)
        section_markers = [
            '##',  # Markdown headers
            '###',
            '####',
            'Chapter',
            'Section',
            'Part',
            'Introduction',
            'Conclusion',
            'Summary'
        ]
        
        def is_section_start(text: str) -> bool:
            """Check if text starts a new section."""
            return any(text.strip().startswith(marker) for marker in section_markers)
        
        def should_split_chunk(paragraph: str) -> bool:
            """Determine if we should split the chunk here."""
            return (
                is_section_start(paragraph) or
                len(paragraph.strip()) > chunk_size * 0.8  # Don't let single paragraphs get too large
            )
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if we should start a new chunk
            if current_chunk and (
                current_length + len(paragraph) > chunk_size or
                should_split_chunk(paragraph)
            ):
                # Join paragraphs and add to chunks
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(paragraph)
            current_length += len(paragraph)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        # Apply smart overlap between chunks
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Get the last paragraph from the previous chunk
                prev_chunk = chunks[i-1]
                prev_paragraphs = prev_chunk.split('\n\n')
                if prev_paragraphs:
                    # Add the last paragraph from previous chunk to maintain context
                    # Only if it's not too long
                    last_para = prev_paragraphs[-1]
                    if len(last_para) <= overlap:
                        final_chunks[-1] = final_chunks[-1] + '\n\n' + last_para
            
            final_chunks.append(chunk)
        
        # Post-process chunks to ensure quality
        processed_chunks = []
        for chunk in final_chunks:
            # Remove excessive whitespace
            chunk = ' '.join(chunk.split())
            # Ensure minimum chunk size
            if len(chunk) >= 100:  # Minimum chunk size
                processed_chunks.append(chunk)
        
        logger.info(f"Completed smart chunking process. Created {len(processed_chunks)} chunks")
        return processed_chunks
    
    async def _generate_embeddings(self, chunks: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generate embeddings for text chunks in parallel batches.
        
        Args:
            chunks: List of text chunks
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of embeddings
        """
        logger.info(f"Starting parallel embedding generation for {len(chunks)} chunks in batches of {batch_size}")
        try:
            embeddings = []
            tasks = []
            
            # Create batches of tasks
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                task = self.embedding_service.generate_embeddings(batch)
                tasks.append(task)
            
            # Process all batches in parallel
            batch_results = await asyncio.gather(*tasks)
            
            # Combine results
            for batch_embeddings in batch_results:
                embeddings.extend(batch_embeddings)
            
            logger.info(f"Successfully generated embeddings for all {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
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