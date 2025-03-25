from pydantic import BaseModel
from typing import Optional
from pathlib import Path

class RedisConfig(BaseModel):
    """Redis configuration settings."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    index_name: str = "embeddings_idx"
    embedding_dimension: int = 768
    chunk_ttl: int = 3600  # 1 hour in seconds

class GoogleDriveConfig(BaseModel):
    """Google Drive configuration settings."""
    credentials_path: str
    root_folder_id: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50

class StorageConfig(BaseModel):
    """Main storage configuration."""
    redis: RedisConfig
    gdrive: GoogleDriveConfig
    base_path: Path = Path("storage")
    cache_enabled: bool = True
    max_cache_size: int = 1000  # Maximum number of chunks to cache

    class Config:
        env_prefix = "STORAGE_" 