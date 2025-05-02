import pytest
from CQE.chunking.semantic_chunker import SemanticChunker

def test_semantic_chunker_initialization():
    """Test that the semantic chunker can be initialized"""
    chunker = SemanticChunker()
    assert chunker is not None

def test_basic_text_chunking():
    """Test basic text chunking with paragraphs"""
    chunker = SemanticChunker()
    text = """
    This is the first paragraph. It contains some text.
    
    This is the second paragraph. It contains different text.
    
    This is the third paragraph. It contains more text.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert "first paragraph" in chunks[0]
    assert "second paragraph" in chunks[1]
    assert "third paragraph" in chunks[2]

def test_minimum_chunk_size():
    """Test that chunks respect minimum size"""
    chunker = SemanticChunker(min_chunk_size=50)
    text = "This is a very short paragraph."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert len(chunks[0]) >= 50

def test_overlap_handling():
    """Test that overlapping chunks are handled correctly"""
    chunker = SemanticChunker(overlap=20)
    text = """
    This is a longer paragraph that will be split into multiple chunks.
    It contains enough text to require multiple chunks.
    The overlap should ensure continuity between chunks.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Check that consecutive chunks have overlap
    for i in range(len(chunks)-1):
        assert len(set(chunks[i][-20:]) & set(chunks[i+1][:20])) > 0
