import pytest
from semantic_chunker import SemanticChunker

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
    print("\n=== Testing basic text chunking ===")
    chunks = chunker.chunk(text)
    print("\nResults:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} (len={len(chunk)}): '{chunk}'")
    assert len(chunks) == 3
    assert "first paragraph" in chunks[0]
    assert "second paragraph" in chunks[1]
    assert "third paragraph" in chunks[2]

def test_minimum_chunk_size():
    """Test that chunks respect minimum size"""
    chunker = SemanticChunker(min_chunk_size=50)
    text = "This is a very short paragraph."
    print("\n=== Testing minimum chunk size ===")
    print(f"Input text (len={len(text)}): '{text}'")
    chunks = chunker.chunk(text)
    print("\nResults:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} (len={len(chunk)}): '{chunk}'")
    assert len(chunks) == 1
    assert len(chunks[0]) >= 50

def test_overlap_handling():
    """Test that overlapping chunks are handled correctly"""
    chunker = SemanticChunker(min_chunk_size=50, overlap=20)
    text = "This is a test sentence that should be split into multiple chunks with proper overlap between them."
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Check that consecutive chunks have overlap
    for i in range(len(chunks)-1):
        overlap = chunks[i][-20:] + chunks[i+1][:20]
        assert len(overlap) > 20  # Should have at least 20 chars of overlap 