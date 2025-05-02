import pytest
from CQE.chunking.semantic_chunker import SECFilingChunker

def test_sec_section_detection():
    """Test that SEC-specific sections are correctly detected"""
    chunker = SECFilingChunker()
    text = """
    ITEM 1. BUSINESS
    This section describes our business operations.
    
    ITEM 1A. RISK FACTORS
    This section outlines potential risks.
    
    PART II
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    This section provides management's perspective.
    
    Table 1. Financial Summary
    This table shows key financial metrics.
    
    Note 1. Accounting Policies
    This note describes our accounting methods.
    """
    sections = chunker.identify_sections(text)
    assert len(sections) == 5
    assert any("ITEM 1. BUSINESS" in section[0] for section in sections)
    assert any("ITEM 1A. RISK FACTORS" in section[0] for section in sections)
    assert any("PART II" in section[0] for section in sections)
    assert any("Table 1." in section[0] for section in sections)
    assert any("Note 1." in section[0] for section in sections)

def test_content_type_detection():
    """Test that SEC content types are correctly detected"""
    chunker = SECFilingChunker()
    
    # Financial content
    financial_text = """
    Consolidated Balance Sheets
    As of December 31, 2023
    Assets: $1,000,000
    Liabilities: $500,000
    """
    assert chunker.detect_content_type(financial_text) == "financial"
    
    # Risk content
    risk_text = """
    Forward-Looking Statements
    This document contains forward-looking statements that involve risks and uncertainties.
    """
    assert chunker.detect_content_type(risk_text) == "risk"
    
    # Business content
    business_text = """
    Business Overview
    Our company operates in the technology sector, competing with major industry players.
    """
    assert chunker.detect_content_type(business_text) == "business"

def test_chunk_preservation():
    """Test that related content stays together in chunks"""
    chunker = SECFilingChunker()
    text = """
    ITEM 1. BUSINESS
    Our company develops software solutions.
    
    Market Position
    We are a leader in our industry segment.
    
    Competition
    We face competition from several major players.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 1  # All business-related content should stay together
    assert "ITEM 1. BUSINESS" in chunks[0]
    assert "Market Position" in chunks[0]
    assert "Competition" in chunks[0]

def test_financial_statement_chunking():
    """Test chunking of financial statements"""
    chunker = SECFilingChunker()
    text = """
    Consolidated Statements of Operations
    For the Year Ended December 31, 2023
    
    Revenue: $1,000,000
    Expenses: $600,000
    Net Income: $400,000
    
    Note 1. Revenue Recognition
    We recognize revenue when control of goods is transferred to customers.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 1  # Financial statement and its note should stay together
    assert "Consolidated Statements" in chunks[0]
    assert "Note 1." in chunks[0]

def test_risk_factor_chunking():
    """Test chunking of risk factors"""
    chunker = SECFilingChunker()
    text = """
    ITEM 1A. RISK FACTORS
    
    Market Risks
    Our business is subject to market fluctuations.
    
    Operational Risks
    We face risks in our day-to-day operations.
    
    Regulatory Risks
    Changes in regulations could impact our business.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 1  # All risk factors should stay together
    assert "ITEM 1A. RISK FACTORS" in chunks[0]
    assert "Market Risks" in chunks[0]
    assert "Operational Risks" in chunks[0]
    assert "Regulatory Risks" in chunks[0]

def test_mda_chunking():
    """Test chunking of Management's Discussion and Analysis"""
    chunker = SECFilingChunker()
    text = """
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    
    Results of Operations
    Our revenue increased by 10% year-over-year.
    
    Liquidity and Capital Resources
    We maintain strong cash reserves.
    
    Critical Accounting Policies
    Our revenue recognition policy is conservative.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 1  # MD&A sections should stay together
    assert "ITEM 7." in chunks[0]
    assert "Results of Operations" in chunks[0]
    assert "Liquidity and Capital Resources" in chunks[0]
    assert "Critical Accounting Policies" in chunks[0]

def test_table_and_note_chunking():
    """Test chunking of tables and their related notes"""
    chunker = SECFilingChunker()
    text = """
    Table 1. Financial Metrics
    Revenue: $1,000,000
    Profit: $200,000
    
    Note 1. Revenue Calculation
    Revenue is calculated based on GAAP principles.
    
    Table 2. Segment Information
    Segment A: $600,000
    Segment B: $400,000
    
    Note 2. Segment Reporting
    Segments are reported based on management structure.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 2  # Each table should stay with its note
    assert "Table 1." in chunks[0] and "Note 1." in chunks[0]
    assert "Table 2." in chunks[1] and "Note 2." in chunks[1]

def test_content_type_edge_cases():
    """Test content type detection with edge cases"""
    chunker = SECFilingChunker()
    
    # Empty text
    assert chunker.detect_content_type("") == "narrative"
    
    # Single word
    assert chunker.detect_content_type("algorithm") == "technical"
    assert chunker.detect_content_type("story") == "narrative"
    
    # Mixed content
    mixed_text = """
    The algorithm was like a magical spell, casting its recursive charm.
    The function danced through the data like a wizard's staff.
    """
    assert chunker.detect_content_type(mixed_text) == "narrative"  # Should default to narrative for mixed content

def test_structure_preservation():
    """Test that document structure is preserved in chunks"""
    chunker = SECFilingChunker()
    text = """
    # Main Heading
    This is the introduction.
    
    ## Subheading 1
    This is the first section.
    
    ### Sub-subheading
    This is a subsection.
    
    ## Subheading 2
    This is the second section.
    """
    chunks = chunker.chunk(text)
    assert any("# Main Heading" in chunk for chunk in chunks)
    assert any("## Subheading 1" in chunk for chunk in chunks)
    assert any("### Sub-subheading" in chunk for chunk in chunks)

def test_structure_preservation_edge_cases():
    """Test structure preservation with edge cases"""
    chunker = SECFilingChunker()
    
    # No headings
    text = "This is a simple paragraph without any headings."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert text in chunks[0]
    
    # Only headings
    text = "# Heading 1\n## Heading 2\n### Heading 3"
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert "# Heading 1" in chunks[0]
    assert "## Heading 2" in chunks[1]
    assert "### Heading 3" in chunks[2]
    
    # Nested sections
    text = """
    # Main
    Content
    ## Sub
    More content
    ### Sub-sub
    Even more content
    ## Another Sub
    Different content
    """
    chunks = chunker.chunk(text)
    assert any("# Main" in chunk and "Content" in chunk for chunk in chunks)
    assert any("## Sub" in chunk and "More content" in chunk for chunk in chunks)

def test_dynamic_chunk_sizing():
    """Test that chunk sizes adapt to content type"""
    chunker = SECFilingChunker()
    technical_text = "Complex algorithm with multiple steps and conditions."
    narrative_text = "A long flowing narrative with descriptive language."
    
    technical_chunks = chunker.chunk(technical_text)
    narrative_chunks = chunker.chunk(narrative_text)
    
    # Technical content should have smaller chunks
    assert len(technical_chunks[0]) < len(narrative_chunks[0])

def test_dynamic_chunk_sizing_edge_cases():
    """Test dynamic chunk sizing with edge cases"""
    chunker = SECFilingChunker(min_chunk_size=50)
    
    # Very short text
    short_text = "Short text."
    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1
    assert len(chunks[0]) >= 50  # Should respect minimum chunk size
    
    # Very long text
    long_text = "Long text. " * 1000
    chunks = chunker.chunk(long_text)
    assert len(chunks) > 1
    assert all(len(chunk) >= 50 for chunk in chunks)
    
    # Mixed content length
    mixed_text = "Short.\n\n" + ("Long paragraph. " * 100) + "\n\nShort again."
    chunks = chunker.chunk(mixed_text)
    assert len(chunks) >= 3

def test_context_preservation():
    """Test that context is preserved across chunks"""
    chunker = SECFilingChunker()
    text = """
    The main character, John, was a skilled programmer.
    He worked at a tech startup in Silicon Valley.
    His latest project involved machine learning algorithms.
    The project was challenging but rewarding.
    """
    chunks = chunker.chunk(text)
    # Check that context (main character's name) is preserved
    assert any("John" in chunk for chunk in chunks)
    # Check that related concepts stay together
    assert any("machine learning" in chunk and "algorithms" in chunk for chunk in chunks)

def test_context_preservation_edge_cases():
    """Test context preservation with edge cases"""
    chunker = SECFilingChunker()
    
    # References across chunks
    text = """
    John introduced himself. He was a programmer.
    Later, John met with his team.
    The team discussed their project with John.
    """
    chunks = chunker.chunk(text)
    assert any("John" in chunk for chunk in chunks)
    
    # Technical terms spread across chunks
    text = """
    The system uses a distributed architecture.
    This architecture enables horizontal scaling.
    Scaling is achieved through load balancing.
    """
    chunks = chunker.chunk(text)
    assert any("distributed" in chunk and "architecture" in chunk for chunk in chunks)
    assert any("scaling" in chunk and "load balancing" in chunk for chunk in chunks)

def test_section_awareness():
    """Test that sections are properly identified and preserved"""
    chunker = SECFilingChunker()
    text = """
    Introduction
    ------------
    This is the introduction section.
    
    Methods
    -------
    This section describes the methods used.
    
    Results
    -------
    This section presents the results.
    """
    chunks = chunker.chunk(text)
    # Check that section headers are preserved with their content
    assert any("Introduction" in chunk and "introduction section" in chunk for chunk in chunks)
    assert any("Methods" in chunk and "methods used" in chunk for chunk in chunks)

def test_section_awareness_edge_cases():
    """Test section awareness with edge cases"""
    chunker = SECFilingChunker()
    
    # Different section formats
    text = """
    # Markdown Style
    Content
    
    1. Numbered Style
    Content
    
    A. Lettered Style
    Content
    
    Underlined Style
    ----------------
    Content
    """
    chunks = chunker.chunk(text)
    assert any("# Markdown Style" in chunk for chunk in chunks)
    assert any("1. Numbered Style" in chunk for chunk in chunks)
    assert any("A. Lettered Style" in chunk for chunk in chunks)
    assert any("Underlined Style" in chunk for chunk in chunks)
    
    # Nested sections
    text = """
    Main Section
    -----------
    Content
    
        Subsection
        ----------
        More content
    
            Sub-subsection
            -------------
            Even more content
    """
    chunks = chunker.chunk(text)
    assert any("Main Section" in chunk and "Content" in chunk for chunk in chunks)
    assert any("Subsection" in chunk and "More content" in chunk for chunk in chunks)

def test_semantic_similarity():
    """Test semantic similarity calculation"""
    chunker = SECFilingChunker()
    
    # Similar content
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleepy dog"
    similarity = chunker.calculate_semantic_similarity(text1, text2)
    assert similarity > 0.5
    
    # Different content
    text3 = "The sun rises in the east"
    text4 = "Programming is fun"
    similarity = chunker.calculate_semantic_similarity(text3, text4)
    assert similarity < 0.3
    
    # Empty or invalid text
    assert chunker.calculate_semantic_similarity("", "text") == 0.0
    assert chunker.calculate_semantic_similarity("text", "") == 0.0
    assert chunker.calculate_semantic_similarity("", "") == 0.0 