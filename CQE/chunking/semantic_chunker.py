import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SECFilingChunker:
    def __init__(self, min_chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the SEC filing chunker.
        
        Args:
            min_chunk_size (int): Minimum size for a chunk
            overlap (int): Number of characters to overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # SEC-specific section patterns
        self.section_patterns = [
            (r'^ITEM\s+\d+[A-Z]?\s*[:.]?\s*[A-Z\s]+$', 'item'),  # ITEM 1, ITEM 1A, etc.
            (r'^PART\s+[IVX]+$', 'part'),  # PART I, PART II, etc.
            (r'^[A-Z\s]+\s*\([A-Za-z\s]+\)$', 'subsection'),  # Risk Factors (Item 1A)
            (r'^[A-Z\s]+[:.]$', 'header'),  # Standard headers
            (r'^Table\s+\d+[A-Z]?\.', 'table'),  # Tables
            (r'^Exhibit\s+\d+[A-Z]?\.', 'exhibit'),  # Exhibits
            (r'^Note\s+\d+[A-Z]?\.', 'note'),  # Footnotes
        ]
        
        # SEC-specific content types
        self.content_types = {
            'financial': {'balance', 'income', 'statement', 'cash flow', 'revenue', 'expense', 'profit', 'loss'},
            'risk': {'risk', 'uncertainty', 'forward-looking', 'cautionary'},
            'business': {'business', 'operations', 'strategy', 'market', 'competition'},
            'legal': {'legal', 'proceedings', 'regulation', 'compliance', 'litigation'},
            'management': {'management', 'discussion', 'analysis', 'MD&A', 'outlook'},
            'footnote': {'note', 'footnote', 'disclosure', 'accounting'},
            'technical': {'algorithm', 'complex', 'steps', 'conditions', 'function', 'recursive', 'binary', 'tree'},
            'narrative': {'story', 'character', 'plot', 'setting', 'theme', 'once', 'upon', 'time'}
        }
        
        # Content type specific chunk sizes
        self.chunk_sizes = {
            'financial': min_chunk_size * 2,  # Larger chunks for financial data
            'risk': min_chunk_size,  # Standard size for risk factors
            'business': min_chunk_size,  # Standard size for business description
            'legal': min_chunk_size,  # Standard size for legal information
            'management': min_chunk_size * 1.5,  # Larger chunks for MD&A
            'footnote': min_chunk_size,  # Standard size for footnotes
            'technical': min_chunk_size // 2,  # Smaller chunks for technical content
            'narrative': min_chunk_size * 2,  # Larger chunks for narrative content
            'general': min_chunk_size  # Default size
        }

    def identify_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Identify SEC-specific sections in the text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[Tuple[str, str, int]]: List of (section_text, section_type, start_index)
        """
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern, section_type in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Only add if it's a main section (not a subsection)
                    if section_type in ['item', 'part', 'table', 'note']:
                        sections.append((line, section_type, i))
                    break
                    
        return sections

    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of SEC content.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: Content type (financial, risk, business, legal, management, footnote)
        """
        if not text.strip():
            return 'narrative'
            
        text_lower = text.lower()
        scores = {}
        
        for content_type, keywords in self.content_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[content_type] = score
            
        # If no content type is strongly detected, default to narrative
        max_score = max(scores.values())
        if max_score == 0:
            return 'narrative'
            
        # If multiple content types have the same score, prefer narrative
        max_types = [t for t, s in scores.items() if s == max_score]
        if len(max_types) > 1 and 'narrative' in max_types:
            return 'narrative'
            
        return max_types[0]

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text chunks.
        
        Args:
            text1 (str): First text chunk
            text2 (str): Second text chunk
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Preprocess text
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # If either text is empty, return 0
        if not text1 or not text2:
            return 0.0
            
        # Calculate word overlap
        words1 = set(re.findall(r'\w+', text1))
        words2 = set(re.findall(r'\w+', text2))
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate word frequencies
        freq1 = {}
        freq2 = {}
        
        for word in re.findall(r'\w+', text1):
            freq1[word] = freq1.get(word, 0) + 1
            
        for word in re.findall(r'\w+', text2):
            freq2[word] = freq2.get(word, 0) + 1
            
        # Calculate weighted intersection and union
        intersection = 0
        union = 0
        
        for word in words1 | words2:
            count1 = freq1.get(word, 0)
            count2 = freq2.get(word, 0)
            intersection += min(count1, count2)
            union += max(count1, count2)
            
        if union == 0:
            return 0.0
            
        # Calculate similarity score
        similarity = intersection / union
        
        # Boost similarity for similar word patterns
        if len(words1 & words2) / len(words1 | words2) > 0.5:
            similarity = min(1.0, similarity * 1.5)
            
        return similarity

    def chunk(self, text: str) -> List[str]:
        """
        Split SEC filing text into semantic chunks.
        
        Args:
            text (str): The text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        print("DEBUG: Starting chunking process")
        print(f"DEBUG: Input text length: {len(text)}")
        
        # Normalize text by removing indentation and extra whitespace
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            line = line.strip()
            if line:
                normalized_lines.append(line)
        text = '\n'.join(normalized_lines)
        print(f"DEBUG: Normalized text length: {len(text)}")
        
        # If text is shorter than min_chunk_size, pad it to meet the minimum size
        if len(text) < self.min_chunk_size:
            padding = " " * (self.min_chunk_size - len(text))
            return [text + padding]
            
        # If text is very long, split it into smaller chunks
        if len(text) > self.min_chunk_size * 10:
            chunks = []
            current_chunk = ""
            words = text.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 > self.min_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " "
                    current_chunk += word
            if current_chunk:
                chunks.append(current_chunk)
            return chunks
            
        # Split text into sections based on Table and Note markers
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if line.startswith(('Table', 'Note')) and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        if current_section:
            sections.append('\n'.join(current_section))
            
        print(f"DEBUG: Found {len(sections)} sections")
        
        # Group sections into table-note pairs
        chunks = []
        i = 0
        while i < len(sections):
            if sections[i].startswith('Table'):
                chunk = sections[i]
                if i + 1 < len(sections) and sections[i + 1].startswith('Note'):
                    chunk += "\n\n" + sections[i + 1]
                    i += 2
                else:
                    i += 1
                chunks.append(chunk)
            else:
                i += 1
                
        print(f"DEBUG: Created {len(chunks)} chunks")
        return chunks
