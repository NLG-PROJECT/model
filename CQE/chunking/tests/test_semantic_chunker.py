import pytest
from CQE.chunking.semantic_chunker import SemanticChunker

def test_financial_statements_grouping():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'Balance Sheet', 'section_type': 'Financial Statements', 'item_number': 'Item 8', 'chunk_type': 'table'},
        {'text': 'Income Statement', 'section_type': 'Financial Statements', 'item_number': 'Item 8', 'chunk_type': 'table'},
    ]
    result = chunker.chunk(chunks)
    assert len(result) == 1
    assert result[0]['section_type'] == 'Financial Statements'
    assert 'Balance Sheet' in result[0]['text']
    assert 'Income Statement' in result[0]['text']
    assert result[0]['chunk_type'] == 'table'

def test_mdna_grouping():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'MD&A intro', 'section_type': 'MD&A', 'item_number': 'Item 7'},
        {'text': 'Liquidity discussion', 'section_type': 'MD&A', 'item_number': 'Item 7'},
    ]
    result = chunker.chunk(chunks)
    assert len(result) == 1
    assert result[0]['section_type'] == 'MD&A'
    assert 'Liquidity discussion' in result[0]['text']
    assert result[0]['chunk_type'] == 'narrative'

def test_risk_factors_chunking():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'Risk Factors Intro', 'section_type': 'Risk Factors', 'item_number': 'Item 1A'},
        {'text': 'Risk 1: Market risk', 'section_type': 'Risk Factors', 'item_number': 'Item 1A'},
        {'text': 'Risk 2: Legal risk', 'section_type': 'Risk Factors', 'item_number': 'Item 1A'},
    ]
    result = chunker.chunk(chunks)
    # Should have intro + each risk as a chunk
    assert any('Intro' in c['text'] for c in result)
    assert any('Market risk' in c['text'] for c in result)
    assert any('Legal risk' in c['text'] for c in result)
    # Each risk chunk should have context from intro
    for c in result:
        if 'Market risk' in c['text'] or 'Legal risk' in c['text']:
            assert 'Intro' in c['context']

def test_footnote_attachment():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'Main content', 'section_type': 'Business Overview', 'item_number': 'Item 1', 'related_sections': ['Note 1']},
        {'text': 'Footnote content', 'section_type': 'Footnote', 'item_number': 'Note 1', 'chunk_type': 'footnote'},
    ]
    result = chunker.chunk(chunks)
    # Main content chunk should have footnotes attached
    main_chunk = next(c for c in result if c['section_type'] == 'Business Overview')
    assert 'footnotes' in main_chunk
    assert any('Footnote content' in f['text'] for f in main_chunk['footnotes'])

def test_table_linking_individual():
    chunker = SemanticChunker()
    # Only test for individual table chunk, not merged
    chunks = [
        {'text': 'Table content', 'section_type': 'Financial Statements', 'item_number': 'Item 8', 'chunk_type': 'table', 'parent_section': 'Item 8'},
        {'text': 'Other content', 'section_type': 'Business Overview', 'item_number': 'Item 1'},
    ]
    # Remove grouping logic for this test
    result = chunker.chunk([chunks[0]])
    table_chunk = result[0]
    assert table_chunk['chunk_type'] == 'table'
    # Should have parent_section as context
    assert table_chunk['parent_section'] == 'Item 8'

def test_metadata_and_hierarchy():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'Some content', 'section_type': 'Business Overview', 'item_number': 'Item 1'},
    ]
    result = chunker.chunk(chunks)
    chunk = result[0]
    assert 'parent_section' in chunk
    assert 'hierarchy' in chunk
    assert chunk['hierarchy'][0] == 'Item 1'

def test_mixed_content():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'MD&A intro', 'section_type': 'MD&A', 'item_number': 'Item 7'},
        {'text': 'Risk Factors Intro', 'section_type': 'Risk Factors', 'item_number': 'Item 1A'},
        {'text': 'Table content', 'section_type': 'Financial Statements', 'item_number': 'Item 8', 'chunk_type': 'table'},
        {'text': 'Footnote content', 'section_type': 'Footnote', 'item_number': 'Note 1', 'chunk_type': 'footnote'},
    ]
    result = chunker.chunk(chunks)
    # Should have at least one chunk for each type
    assert any(c['section_type'] == 'MD&A' for c in result)
    assert any(c['section_type'] == 'Risk Factors' for c in result)
    assert any(c['section_type'] == 'Financial Statements' for c in result)

def test_empty_input():
    chunker = SemanticChunker()
    result = chunker.chunk([])
    assert result == []

def test_unknown_type():
    chunker = SemanticChunker()
    chunks = [
        {'text': 'Unknown content', 'section_type': 'Other', 'item_number': 'Item X'},
    ]
    result = chunker.chunk(chunks)
    assert len(result) == 1
    assert result[0]['section_type'] == 'Other'
