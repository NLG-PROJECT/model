import pytest
import numpy as np
from storage.redis import RedisStorage
import os

@pytest.fixture
def redis_storage():
    """Create a Redis storage instance for testing."""
    storage = RedisStorage(redis_url="redis://localhost:6379/1")  # Use database 1 for testing
    yield storage
    # Cleanup after tests
    storage.client.flushdb()

def test_store_and_get_chunk(redis_storage):
    """Test storing and retrieving a chunk."""
    chunk_id = "test_chunk_1"
    chunk_data = {
        "text": "Test chunk content",
        "embedding": np.array([0.1, 0.2, 0.3]),
        "metadata": {"source": "test"}
    }
    
    # Store chunk
    assert redis_storage.store_chunk(chunk_id, chunk_data) is True
    
    # Retrieve chunk
    retrieved = redis_storage.get_chunk(chunk_id)
    assert retrieved is not None
    assert retrieved["text"] == chunk_data["text"]
    assert np.array_equal(retrieved["embedding"], chunk_data["embedding"])
    assert retrieved["metadata"] == chunk_data["metadata"]

def test_store_and_get_document_metadata(redis_storage):
    """Test storing and retrieving document metadata."""
    doc_id = "test_doc_1"
    metadata = {
        "title": "Test Document",
        "author": "Test Author",
        "created_at": "2024-01-01"
    }
    
    # Store metadata
    assert redis_storage.store_document_metadata(doc_id, metadata) is True
    
    # Retrieve metadata
    retrieved = redis_storage.get_document_metadata(doc_id)
    assert retrieved == metadata

def test_document_chunks(redis_storage):
    """Test managing document chunks."""
    doc_id = "test_doc_2"
    chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
    
    # Store chunks
    for chunk_id in chunk_ids:
        chunk_data = {
            "text": f"Content for {chunk_id}",
            "embedding": np.array([0.1, 0.2, 0.3])
        }
        redis_storage.store_chunk(chunk_id, chunk_data)
        redis_storage.client.sadd(redis_storage._get_doc_chunks_key(doc_id), chunk_id)
    
    # Get chunks
    retrieved_chunks = redis_storage.get_document_chunks(doc_id)
    assert set(retrieved_chunks) == set(chunk_ids)

def test_delete_document(redis_storage):
    """Test deleting a document and its chunks."""
    doc_id = "test_doc_3"
    chunk_id = "chunk_1"
    
    # Store document data
    metadata = {"title": "Test Document"}
    chunk_data = {
        "text": "Test content",
        "embedding": np.array([0.1, 0.2, 0.3])
    }
    
    redis_storage.store_document_metadata(doc_id, metadata)
    redis_storage.store_chunk(chunk_id, chunk_data)
    redis_storage.client.sadd(redis_storage._get_doc_chunks_key(doc_id), chunk_id)
    
    # Delete document
    assert redis_storage.delete_document(doc_id) is True
    
    # Verify deletion
    assert redis_storage.get_document_metadata(doc_id) is None
    assert redis_storage.get_chunk(chunk_id) is None
    assert len(redis_storage.get_document_chunks(doc_id)) == 0

def test_search_similar_chunks(redis_storage):
    """Test searching for similar chunks."""
    # Store some test chunks
    chunks = [
        ("chunk_1", np.array([1.0, 0.0, 0.0])),
        ("chunk_2", np.array([0.0, 1.0, 0.0])),
        ("chunk_3", np.array([0.0, 0.0, 1.0]))
    ]
    
    for chunk_id, embedding in chunks:
        chunk_data = {
            "text": f"Content for {chunk_id}",
            "embedding": embedding
        }
        redis_storage.store_chunk(chunk_id, chunk_data)
    
    # Search with query embedding
    query_embedding = np.array([1.0, 0.0, 0.0])
    results = redis_storage.search_similar_chunks(query_embedding.tolist(), limit=2)
    
    assert len(results) == 2
    assert results[0]["text"] == "Content for chunk_1"  # Should be most similar

def test_health_check(redis_storage):
    """Test health check functionality."""
    assert redis_storage.health_check() is True 