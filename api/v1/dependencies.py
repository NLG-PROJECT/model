from fastapi import Depends
from typing import Generator
import os
from pathlib import Path
from storage.manager import StorageManager
from config.storage import StorageConfig, RedisConfig, GoogleDriveConfig
from .services.file_service import FileService

def get_storage_config() -> StorageConfig:
    """Get storage configuration from environment variables."""
    # Get the absolute path to the credentials file
    credentials_path = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
    if credentials_path:
        # Get the project root directory (3 levels up from this file)
        base_dir = Path(__file__).resolve().parent.parent.parent
        # Resolve the full path
        full_path = base_dir / credentials_path
        credentials_path = str(full_path)

    return StorageConfig(
        redis=RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") if os.getenv("REDIS_PASSWORD") and os.getenv("REDIS_PASSWORD").strip() else None,
            index_name=os.getenv("REDIS_INDEX_NAME", "embeddings_idx"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768"))
        ),
        gdrive=GoogleDriveConfig(
            credentials_path=credentials_path,
            root_folder_id=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID"),
            chunk_size=int(os.getenv("STORAGE_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("STORAGE_CHUNK_OVERLAP", "50"))
        )
    )

def get_storage_manager(config: StorageConfig = Depends(get_storage_config)) -> StorageManager:
    """Get storage manager instance."""
    return StorageManager(config)

def get_file_service(storage_manager: StorageManager = Depends(get_storage_manager)) -> FileService:
    """Get file service instance."""
    return FileService(storage_manager) 