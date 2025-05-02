import re
from typing import List, Dict, Any, Tuple
import logging
import shutil
import os
import json

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    def __init__(self):
        """Initialize the structure analyzer."""
        # Table detection patterns
        self.table_patterns = {
            'header': r'^\s*Table\s+\d+[A-Z]?[\.:]\s*.*$',  # Table headers (e.g., "Table 1: Summary")
            'separator': r'^\s*[\-\+\|\=\s]{3,}\s*$',  # Table separators
            'data_row': r'^\s*\|.*\|\s*$|^\s*[^|]+\t+[^|]+\s*$'  # Table rows with | or tabs
        }
        
        # Footnote patterns
        self.footnote_patterns = {
            'marker': r'^\s*(?:\[\d+\]|\(\d+\)|[*†‡§]|\d+\.)\s+',  # Footnote markers at start of line
            'reference': r'\[\d+\]|\(\d+\)|[*†‡§]'  # References to footnotes
        }
        
        # Cross-reference patterns
        self.cross_ref_patterns = {
            'section': r'(see|refer to|as discussed in)\s+(Section|Item)\s+\d+(\.\d+)?',
            'table': r'(see|as shown in|refer to)\s+Table\s+\d+[A-Z]?',
            'note': r'(see|refer to)\s+Note\s+\d+[A-Z]?'
        }

    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tables from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: List of extracted tables with metadata
        """
        tables = []
        current_table = None
        header_row_found = False
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for table header
            header_match = re.match(self.table_patterns['header'], line)
            if header_match:
                # Save previous table if exists
                if current_table:
                    current_table['end_line'] = i - 1
                    tables.append(current_table)
                
                # Start new table
                current_table = {
                    'title': line.strip(),
                    'start_line': i,
                    'content': [line],
                    'header': [],
                    'rows': [],
                    'footnotes': []
                }
                header_row_found = False
                continue
            
            # If we're in a table
            if current_table:
                current_table['content'].append(line)
                
                # Check if line is a separator
                if re.match(self.table_patterns['separator'], line):
                    continue
                
                # Check if line is a data row
                if re.match(self.table_patterns['data_row'], line):
                    parsed_row = self._parse_table_row(line)
                    if parsed_row:  # Only add non-empty rows
                        if not header_row_found:
                            current_table['header'] = parsed_row
                            header_row_found = True
                        else:
                            current_table['rows'].append(parsed_row)
                else:
                    # Check if line is a footnote
                    footnote_match = re.match(self.footnote_patterns['marker'], line)
                    if footnote_match:
                        current_table['footnotes'].append(line)
                    elif len(current_table['content']) > 1:  # Only end table if we have more than just header
                        # If not a table row or footnote, table ends
                        current_table['end_line'] = i - 1
                        tables.append(current_table)
                        current_table = None
                        header_row_found = False
        
        # Handle last table if exists
        if current_table and len(current_table['content']) > 1:
            current_table['end_line'] = len(lines) - 1
            tables.append(current_table)
        
        return tables

    def _parse_table_row(self, row: str) -> List[str]:
        """Parse a table row into columns."""
        # Skip if row is just separators
        if re.match(r'^[\-\+\|\=\s]+$', row):
            return []
            
        if '|' in row:
            # Split by | and strip whitespace
            return [cell.strip() for cell in row.split('|') if cell.strip()]
        else:
            # Split by tabs
            return [cell.strip() for cell in row.split('\t') if cell.strip()]

    def extract_footnotes(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract footnotes and their references.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing footnotes and references
        """
        footnotes = []
        references = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Find footnote markers at start of line
            footnote_match = re.match(self.footnote_patterns['marker'], line)
            if footnote_match:
                footnote = {
                    'marker': footnote_match.group().strip(),
                    'content': line,
                    'line_number': i
                }
                footnotes.append(footnote)
            
            # Find references to footnotes within text
            for ref_match in re.finditer(self.footnote_patterns['reference'], line):
                # Skip if this is a footnote marker at start of line
                if ref_match.start() == 0 and footnote_match:
                    continue
                    
                reference = {
                    'marker': ref_match.group(),
                    'context': line,
                    'line_number': i,
                    'position': ref_match.span()
                }
                references.append(reference)
        
        return {
            'footnotes': footnotes,
            'references': references
        }

    def map_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Map cross-references in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: List of cross-references with metadata
        """
        cross_refs = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for ref_type, pattern in self.cross_ref_patterns.items():
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    cross_ref = {
                        'type': ref_type,
                        'text': match.group(),
                        'line_number': i,
                        'position': match.span(),
                        'context': line
                    }
                    cross_refs.append(cross_ref)
        
        return cross_refs

    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """
        Perform complete structure analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Complete structure analysis
        """
        # Extract tables
        tables = self.extract_tables(text)
        
        # Extract footnotes
        footnote_info = self.extract_footnotes(text)
        
        # Map cross-references
        cross_refs = self.map_cross_references(text)
        
        # Combine results
        return {
            'tables': tables,
            'footnotes': footnote_info['footnotes'],
            'footnote_references': footnote_info['references'],
            'cross_references': cross_refs,
            'metadata': {
                'table_count': len(tables),
                'footnote_count': len(footnote_info['footnotes']),
                'cross_reference_count': len(cross_refs)
            }
        }

    def extract_and_load_tables(self, pdf_path: str, output_dir: str = "extracted_tables_json", flavor: str = "stream", pages: str = "all") -> list:
        """
        Clear output directory, extract tables from PDF, and load as table chunks with metadata.
        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save JSON files.
            flavor (str): 'stream' for whitespace tables, 'lattice' for bordered tables.
            pages (str): Pages to extract from (e.g., 'all', '1', '1-3').
        Returns:
            List[Dict[str, Any]]: List of table chunks with metadata
        """
        # Clear output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # Extract tables
        import camelot
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
        table_chunks = []
        for i, table in enumerate(tables):
            df = table.df
            # Ensure columns are strings for consistent JSON output
            df.columns = df.columns.map(str)
            json_data = df.to_dict(orient="records")
            json_path = os.path.join(output_dir, f"table_{i+1}.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            # Add as a chunk with metadata
            table_chunks.append({
                'type': 'table',
                'table_index': i+1,
                'json_path': json_path,
                'data': json_data,
                'page': table.page,
                'shape': df.shape
            })
        return table_chunks 