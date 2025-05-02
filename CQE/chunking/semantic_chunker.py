import re
from typing import List, Dict, Any

class SemanticChunker:
    def __init__(self):
        pass

    def chunk(self, analyzed_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk SEC filing content using content-type specific logic and context preservation.
        Args:
            analyzed_chunks (list): List of pre-processed, structured, and analyzed chunks (with metadata)
        Returns:
            list: List of enriched, context-aware chunks
        """
        final_chunks = []
        current_financial = []
        current_mdna = []
        current_risk_intro = None
        current_risk_chunks = []
        footnote_map = {}
        table_map = {}
        parent_section = None

        # First, index footnotes and tables for easy lookup
        for chunk in analyzed_chunks:
            if chunk.get('chunk_type') == 'footnote' or chunk.get('section_type', '').lower() == 'footnote':
                footnote_map[chunk.get('item_number') or chunk.get('header', '')] = chunk
            if chunk.get('chunk_type') == 'table' or chunk.get('section_type', '').lower() == 'financial statements':
                table_map[chunk.get('item_number') or chunk.get('header', '')] = chunk

        for chunk in analyzed_chunks:
            ctype = chunk.get('section_type', '').lower()
            chunk_type = chunk.get('chunk_type', 'narrative')
            # Maintain parent section context
            if chunk.get('item_number'):
                parent_section = chunk.get('item_number')
            chunk['parent_section'] = parent_section

            # Financial Statements: group as one chunk
            if ctype == 'financial statements' or chunk_type == 'table':
                current_financial.append(chunk)
                continue
            # MD&A: larger narrative chunks
            elif ctype == 'md&a':
                current_mdna.append(chunk)
                continue
            # Risk Factors: chunk by individual risk, keep context
            elif ctype == 'risk factors':
                if current_risk_intro is None:
                    current_risk_intro = chunk
                else:
                    current_risk_chunks.append(chunk)
                continue
            # Footnotes: attach to referenced content
            elif ctype == 'footnote' or chunk_type == 'footnote':
                # Will be attached later
                continue
            # Tables: keep as separate, link to context
            elif chunk_type == 'table':
                chunk['linked_context'] = chunk.get('parent_section')
                final_chunks.append(chunk)
                continue
            # Default: narrative or other
            else:
                final_chunks.append(chunk)

        # Group all financial statements as one chunk
        if current_financial:
            merged = self._merge_chunks(current_financial, chunk_type='table', section_type='Financial Statements')
            final_chunks.append(merged)
        # Group all MD&A as one chunk
        if current_mdna:
            merged = self._merge_chunks(current_mdna, chunk_type='narrative', section_type='MD&A')
            final_chunks.append(merged)
        # Risk Factors: intro + each risk as a chunk
        if current_risk_intro:
            final_chunks.append(current_risk_intro)
        for risk_chunk in current_risk_chunks:
            risk_chunk['context'] = current_risk_intro.get('text', '') if current_risk_intro else ''
            final_chunks.append(risk_chunk)
        # Attach footnotes to referenced content
        for chunk in final_chunks:
            related = chunk.get('related_sections', [])
            attached_footnotes = []
            for rel in related:
                if rel in footnote_map:
                    attached_footnotes.append(footnote_map[rel])
            if attached_footnotes:
                chunk['footnotes'] = attached_footnotes
        # Maintain hierarchy and relationships
        for chunk in final_chunks:
            chunk['hierarchy'] = [chunk.get('parent_section'), chunk.get('item_number')]
        return final_chunks

    def _merge_chunks(self, chunks: List[Dict[str, Any]], chunk_type: str, section_type: str) -> Dict[str, Any]:
        """
        Merge a list of chunks into one, preserving metadata and context.
        """
        merged_text = '\n\n'.join([c.get('text', '') for c in chunks])
        merged = {
            'text': merged_text,
            'chunk_type': chunk_type,
            'section_type': section_type,
            'item_number': chunks[0].get('item_number'),
            'parent_section': chunks[0].get('parent_section'),
            'related_sections': sum([c.get('related_sections', []) for c in chunks], []),
            'source_page': chunks[0].get('source_page'),
            'hierarchy': [chunks[0].get('parent_section'), chunks[0].get('item_number')],
        }
        return merged
