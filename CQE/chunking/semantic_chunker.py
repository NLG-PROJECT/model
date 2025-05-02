import re
from typing import List

class SemanticChunker:
    def __init__(self, min_chunk_size: int = 100, overlap: int = 20):
        """
        Initialize the semantic chunker.
        
        Args:
            min_chunk_size (int): Minimum size for a chunk
            overlap (int): Number of characters to overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text (str): The text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # First split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed min_chunk_size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.min_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Handle overlap if needed
        if len(chunks) > 1 and self.overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    # Add overlap from previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-self.overlap:]
                    overlapped_chunks.append(overlap_text + chunks[i])
                else:
                    overlapped_chunks.append(chunks[i])
            chunks = overlapped_chunks
        
        return chunks
