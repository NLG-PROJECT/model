import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base import HardStorageInterface

logger = logging.getLogger(__name__)

class FilesystemStorage(HardStorageInterface):
    """Filesystem-based implementation of hard storage."""
    
    def __init__(self, base_path: str = "storage/files"):
        """Initialize filesystem storage.
        
        Args:
            base_path: Base directory for storing files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.docs_path = self.base_path / "documents"
        self.chunks_path = self.base_path / "chunks"
        self.files_path = self.base_path / "files"
        
        for path in [self.docs_path, self.chunks_path, self.files_path]:
            path.mkdir(exist_ok=True)
    
    async def store_document(self, content: bytes, filename: str, metadata: Dict[str, Any]) -> str:
        """Store a document in the filesystem.
        
        Args:
            content: Document content
            filename: Original filename
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        doc_id = str(datetime.utcnow().timestamp())
        doc_path = self.docs_path / doc_id
        
        # Store document content
        doc_path.mkdir(exist_ok=True)
        with open(doc_path / filename, "wb") as f:
            f.write(content)
        
        # Store metadata
        with open(doc_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        return doc_id
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the filesystem.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        doc_path = self.docs_path / doc_id
        if not doc_path.exists():
            return None
        
        try:
            # Read metadata
            with open(doc_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Read content
            content_file = next(doc_path.glob("*"))
            if content_file.name == "metadata.json":
                content_file = next(doc_path.glob("*"))
            
            with open(content_file, "rb") as f:
                content = f.read()
            
            return {
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    async def store_chunk(self, doc_id: str, chunk_text: str, chunk_metadata: Dict[str, Any]) -> str:
        """Store a document chunk in the filesystem.
        
        Args:
            doc_id: Document ID
            chunk_text: Chunk text content
            chunk_metadata: Chunk metadata
            
        Returns:
            Chunk ID
        """
        chunk_id = str(datetime.utcnow().timestamp())
        chunk_path = self.chunks_path / doc_id / chunk_id
        chunk_path.mkdir(parents=True, exist_ok=True)
        
        # Store chunk content
        with open(chunk_path / "content.txt", "w") as f:
            f.write(chunk_text)
        
        # Store metadata
        with open(chunk_path / "metadata.json", "w") as f:
            json.dump(chunk_metadata, f)
        
        return chunk_id
    
    async def get_chunk(self, doc_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document chunk from the filesystem.
        
        Args:
            doc_id: Document ID
            chunk_id: Chunk ID
            
        Returns:
            Chunk data or None if not found
        """
        chunk_path = self.chunks_path / doc_id / chunk_id
        if not chunk_path.exists():
            return None
        
        try:
            # Read content
            with open(chunk_path / "content.txt", "r") as f:
                content = f.read()
            
            # Read metadata
            with open(chunk_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            return {
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk data
        """
        chunks_path = self.chunks_path / doc_id
        if not chunks_path.exists():
            return []
        
        chunks = []
        for chunk_dir in chunks_path.iterdir():
            if chunk_dir.is_dir():
                chunk_data = await self.get_chunk(doc_id, chunk_dir.name)
                if chunk_data:
                    chunks.append(chunk_data)
        
        return chunks
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete document
            doc_path = self.docs_path / doc_id
            if doc_path.exists():
                for file in doc_path.glob("*"):
                    file.unlink()
                doc_path.rmdir()
            
            # Delete chunks
            chunks_path = self.chunks_path / doc_id
            if chunks_path.exists():
                for chunk_dir in chunks_path.iterdir():
                    if chunk_dir.is_dir():
                        for file in chunk_dir.glob("*"):
                            file.unlink()
                        chunk_dir.rmdir()
                chunks_path.rmdir()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def store_file(self, content: bytes, filename: str) -> str:
        """Store a file in the filesystem.
        
        Args:
            content: File content
            filename: Filename
            
        Returns:
            File ID
        """
        file_id = str(datetime.utcnow().timestamp())
        file_path = self.files_path / file_id
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return file_id
    
    async def get_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve a file from the filesystem.
        
        Args:
            file_id: File ID
            
        Returns:
            File content or None if not found
        """
        file_path = self.files_path / file_id
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if the storage is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if base directories exist and are writable
            for path in [self.docs_path, self.chunks_path, self.files_path]:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = path / ".test"
                test_file.touch()
                test_file.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False 