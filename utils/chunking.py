from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    position: int
    total_chunks: int
    parent_doc_id: str
    created_at: datetime = datetime.now()

class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text according to the strategy."""
        raise NotImplementedError

class RecursiveCharacterChunker(ChunkingStrategy):
    """Chunks text by recursively splitting on characters."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks recursively using separators."""
        chunks = []
        current_pos = 0
        chunk_number = 0
        
        while current_pos < len(text):
            chunk_text = ""
            chunk_start = current_pos
            
            # Try to find the best split point
            for separator in self.separators:
                if separator:
                    next_pos = text.find(separator, current_pos)
                    if next_pos != -1:
                        next_pos += len(separator)
                        if next_pos - current_pos <= self.chunk_size:
                            chunk_text = text[current_pos:next_pos]
                            current_pos = next_pos
                            break
            
            # If no separator found, take a fixed-size chunk
            if not chunk_text:
                chunk_text = text[current_pos:current_pos + self.chunk_size]
                current_pos += self.chunk_size
            
            # Create chunk with metadata
            chunk = Chunk(
                text=chunk_text.strip(),
                metadata={
                    **metadata,
                    "chunk_type": "text",
                    "start_position": chunk_start,
                    "end_position": current_pos
                },
                chunk_id=f"chunk_{chunk_number}",
                position=chunk_number,
                total_chunks=0,  # Will be updated later
                parent_doc_id=metadata.get("doc_id", "unknown")
            )
            chunks.append(chunk)
            chunk_number += 1
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = chunk_number
        
        return chunks

class TableAwareChunker(ChunkingStrategy):
    """Chunks text while preserving table structures."""
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks while preserving table structures."""
        chunks = []
        current_pos = 0
        chunk_number = 0
        
        # Split text into sections (tables and regular text)
        sections = self._split_into_sections(text)
        
        for section in sections:
            if section["type"] == "table":
                # Keep table as a single chunk
                chunk = Chunk(
                    text=section["content"],
                    metadata={
                        **metadata,
                        "chunk_type": "table",
                        "table_id": section.get("table_id", ""),
                        "start_position": current_pos,
                        "end_position": current_pos + len(section["content"])
                    },
                    chunk_id=f"chunk_{chunk_number}",
                    position=chunk_number,
                    total_chunks=0,  # Will be updated later
                    parent_doc_id=metadata.get("doc_id", "unknown")
                )
                chunks.append(chunk)
                current_pos += len(section["content"])
            else:
                # Use recursive chunking for regular text
                text_chunks = RecursiveCharacterChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                ).chunk_text(section["content"], metadata)
                chunks.extend(text_chunks)
                current_pos += len(section["content"])
            
            chunk_number += 1
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = chunk_number
        
        return chunks

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sections of tables and regular text."""
        sections = []
        current_pos = 0
        
        # Find table markers (this is a simplified example)
        table_pattern = r'<table>.*?</table>'
        for match in re.finditer(table_pattern, text, re.DOTALL):
            # Add text before table
            if match.start() > current_pos:
                sections.append({
                    "type": "text",
                    "content": text[current_pos:match.start()]
                })
            
            # Add table
            sections.append({
                "type": "table",
                "content": match.group(0),
                "table_id": f"table_{len(sections)}"
            })
            
            current_pos = match.end()
        
        # Add remaining text
        if current_pos < len(text):
            sections.append({
                "type": "text",
                "content": text[current_pos:]
            })
        
        return sections

def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100
) -> ChunkingStrategy:
    """Factory function to get the appropriate chunking strategy."""
    strategies = {
        "recursive": RecursiveCharacterChunker,
        "table_aware": TableAwareChunker
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return strategies[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    ) 