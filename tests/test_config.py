import pytest
import os
from config import Config, RedisConfig, EmbeddingConfig, ChunkingConfig, load_config

def test_default_config():
    """Test loading configuration with default values."""
    config = load_config()
    
    # Test Redis config
    assert isinstance(config.redis, RedisConfig)
    assert config.redis.url == "redis://localhost:6379"
    assert config.redis.prefix == "docstore"
    
    # Test Embedding config
    assert isinstance(config.embedding, EmbeddingConfig)
    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "llama2"
    assert config.embedding.dimension == 4096
    assert config.embedding.batch_size == 32
    
    # Test Chunking config
    assert isinstance(config.chunking, ChunkingConfig)
    assert config.chunking.chunk_size == 1000
    assert config.chunking.overlap == 100
    assert config.chunking.min_chunk_size == 100
    
    # Test debug mode
    assert config.debug is False

def test_custom_config():
    """Test loading configuration with custom environment variables."""
    # Set custom environment variables
    os.environ["REDIS_URL"] = "redis://custom:6379"
    os.environ["REDIS_PREFIX"] = "custom_prefix"
    os.environ["EMBEDDING_PROVIDER"] = "custom_provider"
    os.environ["EMBEDDING_MODEL"] = "custom_model"
    os.environ["EMBEDDING_DIMENSION"] = "2048"
    os.environ["EMBEDDING_BATCH_SIZE"] = "64"
    os.environ["CHUNK_SIZE"] = "2000"
    os.environ["CHUNK_OVERLAP"] = "200"
    os.environ["MIN_CHUNK_SIZE"] = "200"
    os.environ["DEBUG"] = "true"
    
    try:
        config = load_config()
        
        # Test Redis config
        assert config.redis.url == "redis://custom:6379"
        assert config.redis.prefix == "custom_prefix"
        
        # Test Embedding config
        assert config.embedding.provider == "custom_provider"
        assert config.embedding.model == "custom_model"
        assert config.embedding.dimension == 2048
        assert config.embedding.batch_size == 64
        
        # Test Chunking config
        assert config.chunking.chunk_size == 2000
        assert config.chunking.overlap == 200
        assert config.chunking.min_chunk_size == 200
        
        # Test debug mode
        assert config.debug is True
    finally:
        # Clean up environment variables
        for key in [
            "REDIS_URL", "REDIS_PREFIX", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSION", "EMBEDDING_BATCH_SIZE", "CHUNK_SIZE",
            "CHUNK_OVERLAP", "MIN_CHUNK_SIZE", "DEBUG"
        ]:
            os.environ.pop(key, None)

def test_invalid_config():
    """Test handling of invalid configuration values."""
    # Set invalid environment variables
    os.environ["EMBEDDING_DIMENSION"] = "invalid"
    os.environ["EMBEDDING_BATCH_SIZE"] = "invalid"
    os.environ["CHUNK_SIZE"] = "invalid"
    os.environ["CHUNK_OVERLAP"] = "invalid"
    os.environ["MIN_CHUNK_SIZE"] = "invalid"
    
    try:
        with pytest.raises(ValueError):
            load_config()
    finally:
        # Clean up environment variables
        for key in [
            "EMBEDDING_DIMENSION", "EMBEDDING_BATCH_SIZE", "CHUNK_SIZE",
            "CHUNK_OVERLAP", "MIN_CHUNK_SIZE"
        ]:
            os.environ.pop(key, None) 