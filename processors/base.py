from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
from interfaces.processor import DocumentProcessorInterface

logger = logging.getLogger(__name__)

class BaseProcessor(DocumentProcessorInterface):
    """Base class for document processors with common functionality."""
    
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.max_file_size: int = 50 * 1024 * 1024  # 50MB default
    
    async def validate_document(self, file_path: Path) -> bool:
        """Validate if the document is in the correct format and not corrupted."""
        try:
            # Ensure file_path is a Path object
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file extension: {file_path.suffix}")
                return False
            
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                logger.error(f"File too large: {file_path.stat().st_size} bytes")
                return False
            
            # Check if file exists
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating document {file_path}: {str(e)}")
            return False
    
    def _generate_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Generate basic metadata for the document."""
        return {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size": file_path.stat().st_size,
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """Process a document and return structured data."""
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if not await self.validate_document(file_path):
            raise ValueError(f"Invalid document: {file_path}")
        
        try:
            # Extract all components
            metadata = await self.extract_metadata(file_path)
            text = await self.extract_text(file_path)
            tables = await self.extract_tables(file_path)
            charts = await self.extract_charts(file_path)
            
            # Combine all extracted data
            return {
                "metadata": metadata,
                "content": {
                    "text": text,
                    "tables": tables,
                    "charts": charts
                },
                "processing_status": "success",
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                "metadata": self._generate_metadata(file_path),
                "processing_status": "error",
                "error_message": str(e),
                "processing_timestamp": datetime.utcnow().isoformat()
            } 