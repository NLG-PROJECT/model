import re
from typing import List, Dict, Any
import logging
import unicodedata

logger = logging.getLogger(__name__)

class SECFilingPreprocessor:
    def __init__(self):
        """Initialize the SEC filing preprocessor."""
        # Common SEC filing patterns
        self.section_patterns = {
            'item': r'^ITEM\s+\d+[A-Z]?\s*[:.]?\s*[A-Z\s]+$',
            'part': r'^PART\s+[IVX]+$',
            'table': r'^Table\s+\d+[A-Z]?\.',
            'note': r'^Note\s+\d+[A-Z]?\.',
            'exhibit': r'^Exhibit\s+\d+[A-Z]?\.'
        }
        
        # Subsection patterns
        self.subsection_patterns = {
            'numbered': r'^\d+\.\d+\s+[A-Z]',  # e.g., "1.1 OVERVIEW"
            'lettered': r'^[a-z]\)\s+[A-Z]',   # e.g., "a) Risk Factors"
            'roman': r'^\([ivx]+\)\s+[A-Z]'    # Roman numerals
        }
        
        # Common header/footer patterns
        self.header_footer_patterns = [
            r'^\s*\d+\s*$',  # Page numbers
            r'^\s*[A-Z\s]+\s*$',  # All caps lines
            r'^\s*Form\s+[A-Z0-9-]+\s*$',  # Form numbers
            r'^\s*SEC\s+File\s+No\.\s+\d+-\d+\s*$',  # SEC file numbers
            r'^\s*Page\s+\d+\s*$'  # Page indicators
        ]
        
        # Special character mappings
        self.special_chars = {
            '§': 'Section ',
            '¶': 'Paragraph ',
            '©': '(c)',
            '®': '(R)',
            '™': '(TM)',
            '€': 'EUR ',
            '£': 'GBP ',
            '¥': 'JPY ',
        }
        
        # List markers for preservation
        self.list_markers = [
            '•', '●', '■', '○', '◆', '▪', '▫', '◊',
            *[f"{i}." for i in range(1, 10)],
            *[f"{chr(i)})" for i in range(97, 123)]  # a) to z)
        ]

    def normalize_special_chars(self, text: str) -> str:
        """
        Normalize special characters while preserving meaningful symbols.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized special characters
        """
        # Replace known special characters first
        for char, replacement in self.special_chars.items():
            text = text.replace(char, replacement.strip())
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Handle currency values with proper spacing
        text = re.sub(r'EUR(\d)', r'EUR \1', text)
        text = re.sub(r'JPY(\d)', r'JPY \1', text)
        text = re.sub(r'GBP(\d)', r'GBP \1', text)
        
        # Remove other special characters but keep basic punctuation and bullets
        text = re.sub(r'[^\w\s\.,;:!?\(\)\[\]\{\}\-\'\"/$%•●■○◆▪▫◊]', ' ', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def identify_subsections(self, section_text: str) -> List[Dict[str, Any]]:
        """
        Identify subsections within a section.
        
        Args:
            section_text (str): Text of a section
            
        Returns:
            List[Dict[str, Any]]: List of identified subsections with metadata
        """
        subsections = []
        current_subsection = None
        bullet_points = []
        lines = section_text.split('\n')
        
        def save_current_subsection():
            nonlocal current_subsection, bullet_points, subsections
            if current_subsection:
                if bullet_points:
                    # Add bullet points to the current subsection's content
                    current_subsection['content'].extend(bullet_points)
                    bullet_points = []
                subsections.append(current_subsection)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a bullet point
            is_bullet = any(line.startswith(marker) for marker in ['•', '●', '■', '○', '◆', '▪', '▫', '◊'])
            
            # Check for subsection patterns
            is_new_subsection = False
            subsection_type = None
            
            if not is_bullet:  # Only check for subsection if not a bullet point
                for sub_type, pattern in self.subsection_patterns.items():
                    if re.match(pattern, line):
                        is_new_subsection = True
                        subsection_type = sub_type
                        break
            
            if is_new_subsection:
                # Save previous subsection
                save_current_subsection()
                
                # Start new subsection
                current_subsection = {
                    'type': subsection_type,
                    'title': line,
                    'start_line': i,
                    'content': [line],
                    'level': self._determine_subsection_level(line)
                }
            elif is_bullet:
                bullet_points.append(line)
            else:
                # Regular content line
                if bullet_points and not current_subsection:
                    # If we have bullet points but no subsection, create a bullet subsection
                    subsections.append({
                        'type': 'bullet',
                        'title': bullet_points[0],
                        'start_line': i - len(bullet_points),
                        'content': bullet_points,
                        'level': 1
                    })
                    bullet_points = []
                
                if current_subsection:
                    current_subsection['content'].append(line)
                elif not any(line.startswith(marker) for marker in self.list_markers):
                    # Start a new default subsection for non-list content
                    current_subsection = {
                        'type': 'text',
                        'title': line,
                        'start_line': i,
                        'content': [line],
                        'level': 1
                    }
        
        # Handle any remaining content
        save_current_subsection()
        
        if bullet_points:
            # Create a final bullet subsection if we have leftover bullet points
            subsections.append({
                'type': 'bullet',
                'title': bullet_points[0],
                'start_line': len(lines) - len(bullet_points),
                'content': bullet_points,
                'level': 1
            })
        
        return subsections

    def _determine_subsection_level(self, line: str) -> int:
        """
        Determine the hierarchical level of a subsection.
        
        Args:
            line (str): Subsection header line
            
        Returns:
            int: Hierarchical level (1-based)
        """
        # Count leading spaces to determine indentation level
        indent_level = len(re.match(r'^\s*', line).group())
        
        # Check for numbered subsection patterns
        if re.match(r'^\d+\.\d+', line):
            return 2  # Second level
        elif re.match(r'^\d+\.\d+\.\d+', line):
            return 3  # Third level
        
        # Use indentation as a fallback
        return (indent_level // 4) + 1

    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning for SEC filings.
        
        Args:
            text (str): Raw text from SEC filing
            
        Returns:
            str: Cleaned text
        """
        logger.debug("Starting enhanced text cleaning")
        
        # Split into lines for better control
        lines = text.split('\n')
        cleaned_lines = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_paragraph:
                    cleaned_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Check if line matches any header/footer pattern
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                # Handle special characters
                line = self.normalize_special_chars(line)
                
                # Preserve list formatting
                if any(line.startswith(marker) for marker in self.list_markers):
                    if current_paragraph:
                        cleaned_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    cleaned_lines.append(line)
                else:
                    current_paragraph.append(line)
        
        # Add the last paragraph if exists
        if current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
        
        # Join paragraphs with double newlines
        cleaned_text = '\n\n'.join(cleaned_lines)
        
        logger.debug("Enhanced text cleaning completed")
        return cleaned_text

    def identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Enhanced section identification with subsection support.
        
        Args:
            text (str): Cleaned text from SEC filing
            
        Returns:
            List[Dict[str, Any]]: List of identified sections with metadata
        """
        logger.debug("Starting enhanced section identification")
        sections = []
        current_section = None
        current_subsection = None
        bullet_content = []
        
        def save_bullet_points():
            nonlocal bullet_content, current_subsection, current_section
            if bullet_content:
                if current_subsection:
                    # Add bullet points to current subsection's content
                    current_subsection['content'].extend(bullet_content)
                    # Also create a bullet subsection
                    current_subsection['subsections'].append({
                        'type': 'bullet',
                        'title': bullet_content[0],
                        'start_line': current_subsection['start_line'] + len(current_subsection['content']),
                        'content': bullet_content.copy(),
                        'level': current_subsection['level'] + 1
                    })
                else:
                    # Add bullet points to current section's content
                    current_section['content'].extend(bullet_content)
                    # Also create a bullet subsection
                    current_section['subsections'].append({
                        'type': 'bullet',
                        'title': bullet_content[0],
                        'start_line': current_section['start_line'] + len(current_section['content']),
                        'content': bullet_content.copy(),
                        'level': 1
                    })
                bullet_content = []
        
        def save_current_subsection():
            nonlocal current_section, current_subsection
            if current_subsection:
                save_bullet_points()
                current_section['subsections'].append(current_subsection)
                current_subsection = None
        
        def is_bullet_point(line):
            return any(line.startswith(marker) for marker in ['•', '●', '■', '○', '◆', '▪', '▫', '◊'])
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check for main section patterns first
            is_main_section = False
            for section_type, pattern in self.section_patterns.items():
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section if exists
                    if current_section:
                        save_current_subsection()
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'type': section_type,
                        'title': line,
                        'start_line': i,
                        'content': [line],
                        'subsections': []
                    }
                    current_subsection = None
                    bullet_content = []
                    is_main_section = True
                    break
            
            if is_main_section:
                i += 1
                continue
            
            # If we're not in a section yet, skip
            if not current_section:
                i += 1
                continue
            
            # Check for subsection patterns
            is_subsection = False
            for sub_type, pattern in self.subsection_patterns.items():
                if re.match(pattern, line, re.IGNORECASE):
                    # Save current subsection if exists
                    save_current_subsection()
                    
                    # Start new subsection
                    current_subsection = {
                        'type': sub_type,
                        'title': line,
                        'start_line': i,
                        'content': [line],
                        'level': self._determine_subsection_level(line),
                        'subsections': []
                    }
                    bullet_content = []
                    is_subsection = True
                    break
            
            if is_subsection:
                i += 1
                continue
            
            # Handle bullet points and regular content
            if is_bullet_point(line):
                bullet_content.append(line)
            else:
                # Save any accumulated bullet points before adding regular content
                save_bullet_points()
                # Regular content
                if current_subsection:
                    current_subsection['content'].append(line)
                else:
                    current_section['content'].append(line)
            
            i += 1
        
        # Handle any remaining content
        if current_section:
            save_current_subsection()
            sections.append(current_section)
        
        logger.debug(f"Identified {len(sections)} sections")
        return sections

    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Enhanced preprocessing function for SEC filings.
        
        Args:
            text (str): Raw text from SEC filing
            
        Returns:
            Dict[str, Any]: Preprocessed text with metadata
        """
        logger.info("Starting enhanced SEC filing preprocessing")
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Identify sections and subsections
        sections = self.identify_sections(cleaned_text)
        
        # Count total subsections and collect all subsection types
        total_subsections = 0
        subsection_types = set()
        
        def process_subsection(subsection):
            nonlocal total_subsections
            subsection_types.add(subsection['type'])
            total_subsections += 1
            # Process nested subsections if any
            if 'subsections' in subsection:
                for nested in subsection['subsections']:
                    process_subsection(nested)
        
        # Process all sections and their subsections
        for section in sections:
            for subsection in section['subsections']:
                process_subsection(subsection)
        
        # Prepare the result
        result = {
            'cleaned_text': cleaned_text,
            'sections': sections,
            'metadata': {
                'total_sections': len(sections),
                'total_subsections': total_subsections,
                'section_types': list(set(s['type'] for s in sections)),
                'subsection_types': list(subsection_types)
            }
        }
        
        logger.info("Enhanced SEC filing preprocessing completed")
        return result 