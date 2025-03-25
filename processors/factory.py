from pathlib import Path
from typing import Dict, Type
import logging
from .base import BaseProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .excel_processor import ExcelProcessor

logger = logging.getLogger(__name__)

class ProcessorFactory:
    """Factory class for creating document processors."""
    
    _processors: Dict[str, Type[BaseProcessor]] = {
        '.pdf': PDFProcessor,
        '.docx': DOCXProcessor,
        '.xlsx': ExcelProcessor,
        '.xls': ExcelProcessor
    }
    
    @classmethod
    def get_processor(cls, file_path: Path) -> BaseProcessor:
        """Get the appropriate processor for the file type."""
        extension = file_path.suffix.lower()
        logger.info(f"Getting processor for file: {file_path}")
        logger.info(f"File extension: {extension}")
        logger.info(f"Supported extensions: {cls.get_supported_extensions()}")
        
        if extension not in cls._processors:
            error_msg = f"No processor available for file type: {extension}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        processor = cls._processors[extension]()
        logger.info(f"Created processor: {processor.__class__.__name__}")
        return processor
    
    @classmethod
    def register_processor(cls, extension: str, processor_class: Type[BaseProcessor]):
        """Register a new processor for a file type."""
        extension = extension.lower()
        cls._processors[extension] = processor_class
        logger.info(f"Registered processor {processor_class.__name__} for extension {extension}")
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """Get list of supported file extensions."""
        return list(cls._processors.keys()) 