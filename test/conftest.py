import pytest
import os
import tempfile
from pathlib import Path
from typing import Generator
import asyncio
from fastapi import UploadFile
from api.v1.services.file_service import FileService
from storage.manager import StorageManager
from config.storage import StorageConfig, RedisConfig, GoogleDriveConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return StorageConfig(
        redis=RedisConfig(
            host=os.getenv("TEST_REDIS_HOST", "localhost"),
            port=int(os.getenv("TEST_REDIS_PORT", "6379")),
            password=os.getenv("TEST_REDIS_PASSWORD", ""),
            index_name="test_embeddings_idx",
            embedding_dimension=768
        ),
        gdrive=GoogleDriveConfig(
            credentials_path=os.getenv("TEST_GOOGLE_DRIVE_CREDENTIALS_PATH"),
            root_folder_id=os.getenv("TEST_GOOGLE_DRIVE_ROOT_FOLDER_ID"),
            chunk_size=500,
            chunk_overlap=50
        )
    )

@pytest.fixture(scope="session")
async def storage_manager(test_config):
    """Provide a storage manager instance for testing."""
    manager = StorageManager(test_config)
    yield manager
    # Cleanup after tests
    await manager.flush_redis()

@pytest.fixture(scope="session")
async def file_service(test_config):
    """Provide a file service instance for testing."""
    service = FileService()
    yield service
    # Cleanup after tests
    await service.storage_manager.flush_redis()

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="function")
def sample_files(temp_dir) -> Generator[list[Path], None, None]:
    """Create sample test files."""
    files = []
    
    # Create a sample PDF
    pdf_path = temp_dir / "test.pdf"
    # TODO: Create a sample PDF file
    files.append(pdf_path)
    
    # Create a sample DOCX
    docx_path = temp_dir / "test.docx"
    # TODO: Create a sample DOCX file
    files.append(docx_path)
    
    # Create a sample Excel file
    excel_path = temp_dir / "test.xlsx"
    # TODO: Create a sample Excel file
    files.append(excel_path)
    
    yield files

@pytest.fixture(scope="function")
def upload_files(sample_files) -> list[UploadFile]:
    """Create UploadFile objects from sample files."""
    return [
        UploadFile(
            filename=file.name,
            file=open(file, "rb"),
            content_type=f"application/{file.suffix[1:]}"
        )
        for file in sample_files
    ] 