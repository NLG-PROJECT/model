from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

class DocumentProcessorInterface(ABC):
    """Base interface for document processors."""
    
    @abstractmethod
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process a document and return structured data."""
        pass
    
    @abstractmethod
    async def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from the document."""
        pass
    
    @abstractmethod
    async def extract_text(self, file_path: Path) -> str:
        """Extract text content from the document."""
        pass
    
    @abstractmethod
    async def extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from the document."""
        pass
    
    @abstractmethod
    async def extract_charts(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract charts/graphs from the document."""
        pass
    
    @abstractmethod
    async def validate_document(self, file_path: Path) -> bool:
        """Validate if the document is in the correct format and not corrupted."""
        pass 