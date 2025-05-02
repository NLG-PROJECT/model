import pytest
from ..CQE.chunking.structure_analyzer import StructureAnalyzer

def test_table_extraction():
    """Test table extraction functionality."""
    analyzer = StructureAnalyzer()
    
    input_text = """
    Some text before the table.
    
    Table 1: Financial Summary
    |-------------------|
    | Year | Revenue    |
    |-------------------|
    | 2020 | $1,000,000 |
    | 2021 | $1,500,000 |
    |-------------------|
    [1] All figures in USD
    
    Some text after the table.
    """
    
    tables = analyzer.extract_tables(input_text)
    
    assert len(tables) == 1
    assert tables[0]['title'] == 'Table 1: Financial Summary'
    assert len(tables[0]['rows']) == 2
    assert tables[0]['rows'][0] == ['2020', '$1,000,000']
    assert tables[0]['footnotes'] == ['[1] All figures in USD']

def test_footnote_extraction():
    """Test footnote extraction functionality."""
    analyzer = StructureAnalyzer()
    
    input_text = """
    This is some text with a footnote reference[1] and another one[2].
    There's also a different style footnote* and another†.
    
    [1] First footnote explanation
    [2] Second footnote explanation
    * Alternative footnote style
    † Another alternative style
    """
    
    footnote_info = analyzer.extract_footnotes(input_text)
    
    assert len(footnote_info['footnotes']) == 4
    assert len(footnote_info['references']) == 4
    
    # Verify footnote content
    footnote_texts = [f['content'] for f in footnote_info['footnotes']]
    assert '[1] First footnote explanation' in footnote_texts
    assert '* Alternative footnote style' in footnote_texts
    
    # Verify references
    ref_markers = [r['marker'] for r in footnote_info['references']]
    assert '[1]' in ref_markers
    assert '*' in ref_markers

def test_cross_reference_mapping():
    """Test cross-reference mapping functionality."""
    analyzer = StructureAnalyzer()
    
    input_text = """
    As discussed in Section 2.1, our financial performance improved.
    Please refer to Table 3 for detailed metrics.
    See Note 1 for accounting policies.
    """
    
    cross_refs = analyzer.map_cross_references(input_text)
    
    assert len(cross_refs) == 3
    
    # Verify reference types
    ref_types = [ref['type'] for ref in cross_refs]
    assert 'section' in ref_types
    assert 'table' in ref_types
    assert 'note' in ref_types
    
    # Verify reference text
    ref_texts = [ref['text'] for ref in cross_refs]
    assert any('Section 2.1' in text for text in ref_texts)
    assert any('Table 3' in text for text in ref_texts)
    assert any('Note 1' in text for text in ref_texts)

def test_complete_structure_analysis():
    """Test complete structure analysis functionality."""
    analyzer = StructureAnalyzer()
    
    input_text = """
    As shown in Table 1, our revenue increased.
    
    Table 1: Revenue Growth
    |---------|------------|
    | Year    | Revenue    |
    |---------|------------|
    | 2020[1] | $1,000,000 |
    | 2021    | $1,500,000 |
    |---------|------------|
    [1] Impacted by COVID-19
    
    See Section 3.2 for detailed analysis.
    """
    
    result = analyzer.analyze_structure(input_text)
    
    # Verify all components are present
    assert 'tables' in result
    assert 'footnotes' in result
    assert 'footnote_references' in result
    assert 'cross_references' in result
    assert 'metadata' in result
    
    # Verify metadata
    assert result['metadata']['table_count'] == 1
    assert result['metadata']['footnote_count'] == 1
    assert result['metadata']['cross_reference_count'] == 2  # "Table 1" and "Section 3.2"
    
    # Verify table content
    assert len(result['tables']) == 1
    assert 'Revenue Growth' in result['tables'][0]['title']
    
    # Verify footnotes
    assert any('COVID-19' in f['content'] for f in result['footnotes'])
    
    # Verify cross-references
    ref_texts = [ref['text'] for ref in result['cross_references']]
    assert any('Section 3.2' in text for text in ref_texts) 