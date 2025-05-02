import pytest
from ..preprocessor import SECFilingPreprocessor

def test_clean_text():
    """Test basic text cleaning functionality."""
    preprocessor = SECFilingPreprocessor()
    
    # Test input with various issues
    input_text = """
    Page 1
    
    SEC File No. 123-456
    
    ITEM 1. BUSINESS
    
    This is a test paragraph.
    
    Page 2
    
    ITEM 1A. RISK FACTORS
    
    Another test paragraph.
    """
    
    cleaned_text = preprocessor.clean_text(input_text)
    
    # Verify cleaning results
    assert "Page 1" not in cleaned_text
    assert "SEC File No. 123-456" not in cleaned_text
    assert "ITEM 1. BUSINESS" in cleaned_text
    assert "ITEM 1A. RISK FACTORS" in cleaned_text
    assert cleaned_text.count("\n\n") == 3  # Three paragraph breaks: between ITEM 1 and its content, between paragraphs, and between ITEM 1A and its content

def test_special_character_handling():
    """Test special character normalization."""
    preprocessor = SECFilingPreprocessor()
    
    input_text = """
    § 123 Legal Notice
    Company® and Product™
    Cost: €100 or ¥10000
    • First bullet point
    ○ Second bullet point
    a) First item
    b) Second item
    """
    
    cleaned_text = preprocessor.normalize_special_chars(input_text)
    
    # Verify special character handling
    assert "Section 123" in cleaned_text
    assert "Company(R)" in cleaned_text
    assert "Product(TM)" in cleaned_text
    assert "EUR 100" in cleaned_text
    assert "JPY 10000" in cleaned_text
    assert "• First" in cleaned_text  # List markers preserved
    assert "○ Second" in cleaned_text
    assert "a) First" in cleaned_text
    assert "b) Second" in cleaned_text

def test_subsection_identification():
    """Test subsection identification functionality."""
    preprocessor = SECFilingPreprocessor()
    
    input_text = """
    ITEM 1. BUSINESS

    1.1 COMPANY OVERVIEW
    Our company focuses on technology.

    1.2 MARKET OPPORTUNITY
    The market is growing.

    a) Market Size
    The total addressable market.

    b) Growth Trends
    Key trends include:
    • AI adoption
    • Cloud computing
    • Mobile technology
    """
    
    sections = preprocessor.identify_sections(input_text)
    
    # Verify section and subsection identification
    assert len(sections) == 1
    assert sections[0]['type'] == 'item'
    assert sections[0]['title'] == 'ITEM 1. BUSINESS'
    
    subsections = sections[0]['subsections']
    assert len(subsections) == 4  # 1.1, 1.2, a), b)
    
    # Verify subsection content
    assert subsections[0]['type'] == 'numbered'
    assert subsections[0]['title'] == '1.1 COMPANY OVERVIEW'
    assert 'Our company focuses on technology.' in subsections[0]['content']
    
    assert subsections[1]['type'] == 'numbered'
    assert subsections[1]['title'] == '1.2 MARKET OPPORTUNITY'
    assert 'The market is growing.' in subsections[1]['content']
    
    assert subsections[2]['type'] == 'lettered'
    assert subsections[2]['title'] == 'a) Market Size'
    assert 'The total addressable market.' in subsections[2]['content']
    
    assert subsections[3]['type'] == 'lettered'
    assert subsections[3]['title'] == 'b) Growth Trends'
    assert 'Key trends include:' in subsections[3]['content']
    assert '• AI adoption' in subsections[3]['content']
    assert '• Cloud computing' in subsections[3]['content']
    assert '• Mobile technology' in subsections[3]['content']

def test_identify_sections():
    """Test section identification functionality."""
    preprocessor = SECFilingPreprocessor()
    
    # Test input with multiple sections
    input_text = """
    ITEM 1. BUSINESS
    
    Business description here.
    
    ITEM 1A. RISK FACTORS
    
    Risk factors description.
    
    PART II
    
    ITEM 7. MANAGEMENT'S DISCUSSION
    """
    
    sections = preprocessor.identify_sections(input_text)
    
    # Verify section identification
    assert len(sections) == 3
    assert sections[0]['type'] == 'item'
    assert sections[0]['title'] == 'ITEM 1. BUSINESS'
    assert sections[1]['type'] == 'item'
    assert sections[1]['title'] == 'ITEM 1A. RISK FACTORS'
    assert sections[2]['type'] == 'part'
    assert sections[2]['title'] == 'PART II'

def test_preprocess():
    """Test complete preprocessing functionality."""
    preprocessor = SECFilingPreprocessor()
    
    # Test input with various elements
    input_text = """
    Page 1
    
    SEC File No. 123-456
    
    ITEM 1. BUSINESS
    
    1.1 OVERVIEW
    Business description here.
    
    • Key point 1
    • Key point 2
    
    Page 2
    
    ITEM 1A. RISK FACTORS
    
    a) Market Risk
    Risk description here.
    
    b) Technology Risk
    • Security concerns
    • Integration issues
    """
    
    result = preprocessor.preprocess(input_text)
    
    # Verify preprocessing results
    assert 'cleaned_text' in result
    assert 'sections' in result
    assert 'metadata' in result
    assert result['metadata']['total_sections'] == 2
    assert result['metadata']['total_subsections'] > 0
    assert 'item' in result['metadata']['section_types']
    assert set(result['metadata']['subsection_types']) == {'numbered', 'lettered'}
    
    # Verify bullet points are included in their parent subsection's content
    business_section = result['sections'][0]
    overview_subsection = business_section['subsections'][0]
    assert any('Key point 1' in content for content in overview_subsection['content'])
    assert any('Key point 2' in content for content in overview_subsection['content'])
    
    risk_section = result['sections'][1]
    tech_risk_subsection = risk_section['subsections'][-1]
    assert any('Security concerns' in content for content in tech_risk_subsection['content'])
    assert any('Integration issues' in content for content in tech_risk_subsection['content']) 