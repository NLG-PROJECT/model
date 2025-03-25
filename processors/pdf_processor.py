from pathlib import Path
from typing import Dict, Any, List
import logging
from PyPDF2 import PdfReader
import io
from processors.base import BaseProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(BaseProcessor):
    """Processor for PDF documents."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']
    
    async def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        try:
            metadata = self._generate_metadata(file_path)
            
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                pdf_metadata = pdf.metadata
                
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get('/Title', ''),
                        "author": pdf_metadata.get('/Author', ''),
                        "subject": pdf_metadata.get('/Subject', ''),
                        "keywords": pdf_metadata.get('/Keywords', ''),
                        "creator": pdf_metadata.get('/Creator', ''),
                        "producer": pdf_metadata.get('/Producer', ''),
                        "page_count": len(pdf.pages)
                    })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return self._generate_metadata(file_path)
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text content from PDF document."""
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"Page {page_num}:\n{text}\n")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        continue
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    async def extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF document."""
        # Note: This is a placeholder. For actual table extraction,
        # you would need to use a more sophisticated library like camelot-py or tabula-py
        try:
            tables = []
            # TODO: Implement table extraction using camelot-py or tabula-py
            return tables
        except Exception as e:
            logger.error(f"Error extracting PDF tables: {str(e)}")
            return []
    
    async def extract_charts(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract charts/graphs from PDF document."""
        # Note: This is a placeholder. For actual chart extraction,
        # you would need to use image processing libraries
        try:
            charts = []
            # TODO: Implement chart extraction using image processing
            return charts
        except Exception as e:
            logger.error(f"Error extracting PDF charts: {str(e)}")
            return [] 