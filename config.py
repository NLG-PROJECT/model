import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class RedisConfig:
    """Redis configuration settings."""
    url: str
    prefix: str = "docstore"
    use_ssl: bool = False
    username: Optional[str] = None
    password: Optional[str] = None

@dataclass
class EmbeddingConfig:
    """Embedding service configuration settings."""
    provider: str
    model: str
    dimension: int
    batch_size: int = 32

@dataclass
class ChunkingConfig:
    """Text chunking configuration settings."""
    chunk_size: int
    overlap: int
    min_chunk_size: int = 100

@dataclass
class Config:
    """Main configuration class."""
    redis: RedisConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    debug: bool = False

def load_config() -> Config:
    """Load configuration from environment variables."""
    # Redis configuration
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_prefix = os.getenv("REDIS_PREFIX", "docstore")
    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    redis_use_ssl = os.getenv("REDIS_USE_SSL", "false").lower() == "true"
    
    # Embedding configuration
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    embedding_model = os.getenv("EMBEDDING_MODEL", "llama2")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "4096"))
    embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Chunking configuration
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    
    # Debug mode
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    return Config(
        redis=RedisConfig(
            url=redis_url,
            prefix=redis_prefix,
            use_ssl=redis_use_ssl,
            username=redis_username,
            password=redis_password
        ),
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            dimension=embedding_dimension,
            batch_size=embedding_batch_size
        ),
        chunking=ChunkingConfig(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        ),
        debug=debug
    )

# Global configuration instance
config = load_config() 