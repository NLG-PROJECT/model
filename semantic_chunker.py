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
        # Clean text and split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        print(f"Found {len(paragraphs)} paragraphs")
        
        # For short text, pad it
        if len(paragraphs) == 1 and len(paragraphs[0]) < self.min_chunk_size:
            result = paragraphs[0] + " " * (self.min_chunk_size - len(paragraphs[0]))
            return [result]
        
        # For longer text, normalize spaces within paragraphs
        paragraphs = [' '.join(p.split()) for p in paragraphs]
        
        # If we have multiple paragraphs, return them as chunks
        if len(paragraphs) > 1:
            chunks = []
            for p in paragraphs:
                if len(p) < self.min_chunk_size:
                    p = p + " " * (self.min_chunk_size - len(p))
                chunks.append(p)
            return chunks
        
        # If we have one long paragraph, split it with overlap
        text = paragraphs[0]
        chunks = []
        pos = 0
        
        while pos < len(text):
            # Calculate chunk boundaries
            end = min(pos + self.min_chunk_size, len(text))
            chunk = text[pos:end]
            
            # Pad if necessary
            if len(chunk) < self.min_chunk_size:
                chunk = chunk + " " * (self.min_chunk_size - len(chunk))
            
            chunks.append(chunk)
            
            # Move to next position, ensuring we don't go backwards
            next_pos = end - self.overlap
            if next_pos <= pos:
                break  # Prevent infinite loop
            pos = next_pos
        
        return chunks 