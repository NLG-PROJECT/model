import yaml
import re
from typing import List, Dict, Any
import os

class ContentAnalyzer:
    def __init__(self, schema_path: str = "sec_content_schema.yaml"):
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        with open(schema_path, "r") as f:
            self.schema = yaml.safe_load(f)
        self.content_types = self.schema.get("content_types", {})
        self.item_number_mapping = self.schema.get("item_number_mapping", {})
        self.relationship_patterns = self.schema.get("relationship_patterns", [])

    def classify_content_type(self, text: str) -> str:
        """Classify the content type of a chunk based on schema patterns."""
        for ctype, patterns in self.content_types.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ctype
        return "Other"

    def map_item_number(self, header: str) -> str:
        """Map a section header to a standard item number using the schema."""
        # Sort item numbers by length descending to match more specific items first
        for item in sorted(self.item_number_mapping.keys(), key=len, reverse=True):
            if re.search(item, header, re.IGNORECASE):
                return item
        return "Unknown"

    def detect_relationships(self, text: str) -> List[str]:
        """Detect references/relationships in the text using schema patterns."""
        found = []
        for pattern in self.relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.extend(matches)
        return found

    def analyze_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a chunk, classify its type, map item number, detect relationships, and attach metadata.
        Args:
            chunk (dict): Chunk with at least 'text' and/or 'header'.
        Returns:
            dict: Chunk with added metadata.
        """
        text = chunk.get('text', '')
        header = chunk.get('header', text)
        chunk['section_type'] = self.classify_content_type(header + '\n' + text)
        chunk['item_number'] = self.map_item_number(header)
        chunk['related_sections'] = self.detect_relationships(text)
        return chunk

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a list of chunks, attaching metadata to each.
        Args:
            chunks (list): List of chunk dicts.
        Returns:
            list: List of enriched chunk dicts.
        """
        return [self.analyze_chunk(chunk) for chunk in chunks] 