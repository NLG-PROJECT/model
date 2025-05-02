import pytest
from ..content_analyzer import ContentAnalyzer

def test_classify_content_type():
    analyzer = ContentAnalyzer("sec_content_schema.yaml")
    assert analyzer.classify_content_type("Consolidated Balance Sheets") == "Financial Statements"
    assert analyzer.classify_content_type("Liquidity and Capital Resources") == "MD&A"
    assert analyzer.classify_content_type("General macro risks") == "Risk Factors"
    assert analyzer.classify_content_type("Unknown section") == "Other"

def test_map_item_number():
    analyzer = ContentAnalyzer("sec_content_schema.yaml")
    assert analyzer.map_item_number("Item 1A. Risk Factors") == "Item 1A"
    assert analyzer.map_item_number("Item 3. Legal Proceedings") == "Item 3"
    assert analyzer.map_item_number("Random Header") == "Unknown"

def test_detect_relationships():
    analyzer = ContentAnalyzer("sec_content_schema.yaml")
    text = "See Note 2 to the Consolidated Financial Statements. As discussed in MD&A."
    rels = analyzer.detect_relationships(text)
    assert any("Note 2" in r or "MD&A" in r for r in rels)

def test_analyze_chunk():
    analyzer = ContentAnalyzer("sec_content_schema.yaml")
    chunk = {
        'header': 'Item 1A. Risk Factors',
        'text': 'See Note 2 to the Consolidated Financial Statements.'
    }
    enriched = analyzer.analyze_chunk(chunk)
    assert enriched['section_type'] == "Risk Factors"
    assert enriched['item_number'] == "Item 1A"
    assert any("Note 2" in r for r in enriched['related_sections'])

def test_analyze_chunks():
    analyzer = ContentAnalyzer("sec_content_schema.yaml")
    chunks = [
        {'header': 'Item 1. Business', 'text': 'Strategy and market overview.'},
        {'header': 'Item 1A. Risk Factors', 'text': 'General macro risks.'},
        {'header': 'Item 8. Financial Statements', 'text': 'Consolidated Balance Sheets.'}
    ]
    enriched_chunks = analyzer.analyze_chunks(chunks)
    assert enriched_chunks[0]['section_type'] == "Business Overview"
    assert enriched_chunks[1]['section_type'] == "Risk Factors"
    assert enriched_chunks[2]['section_type'] == "Financial Statements" 