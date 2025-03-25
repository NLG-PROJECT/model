import pytest
from pathlib import Path
from fastapi import UploadFile
from api.v1.services.file_service import FileService

@pytest.mark.asyncio
async def test_process_files(file_service, upload_files):
    """Test processing multiple files."""
    results = await file_service.process_files(upload_files)
    
    # Verify results
    assert len(results) == len(upload_files)
    for result in results:
        assert result["status"] == "success"
        assert "file_id" in result
        assert "chunks_count" in result
        assert result["chunks_count"] > 0

@pytest.mark.asyncio
async def test_get_file_info(file_service, upload_files):
    """Test retrieving file information."""
    # First process a file
    results = await file_service.process_files([upload_files[0]])
    file_id = results[0]["file_id"]
    
    # Get file info
    file_info = await file_service.get_file_info(file_id)
    
    # Verify file info
    assert file_info is not None
    assert file_info["file_id"] == file_id
    assert file_info["filename"] == upload_files[0].filename
    assert "content_type" in file_info
    assert "size" in file_info
    assert "created_at" in file_info
    assert "metadata" in file_info

@pytest.mark.asyncio
async def test_list_files(file_service, upload_files):
    """Test listing all files."""
    # Process some files first
    await file_service.process_files(upload_files)
    
    # List files
    files = await file_service.list_files()
    
    # Verify files list
    assert len(files) >= len(upload_files)
    for file in files:
        assert "file_id" in file
        assert "filename" in file
        assert "content_type" in file
        assert "size" in file
        assert "created_at" in file
        assert "metadata" in file

@pytest.mark.asyncio
async def test_delete_file(file_service, upload_files):
    """Test deleting a file."""
    # First process a file
    results = await file_service.process_files([upload_files[0]])
    file_id = results[0]["file_id"]
    
    # Delete the file
    success = await file_service.delete_file(file_id)
    assert success is True
    
    # Verify file is deleted
    file_info = await file_service.get_file_info(file_id)
    assert file_info is None

@pytest.mark.asyncio
async def test_error_handling(file_service):
    """Test error handling with invalid files."""
    # Create an invalid file
    invalid_file = UploadFile(
        filename="invalid.txt",
        file=None,
        content_type="text/plain"
    )
    
    # Process invalid file
    results = await file_service.process_files([invalid_file])
    
    # Verify error handling
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "error" in results[0]

@pytest.mark.asyncio
async def test_chunking_and_embedding(file_service, upload_files):
    """Test chunking and embedding generation."""
    # Process a file
    results = await file_service.process_files([upload_files[0]])
    file_id = results[0]["file_id"]
    
    # Get file info
    file_info = await file_service.get_file_info(file_id)
    
    # Verify chunks were created
    assert file_info["chunks_count"] > 0
    
    # Verify embeddings were generated
    chunks = await file_service.storage_manager.get_document_chunks(file_id)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "embedding" in chunk
        assert len(chunk["embedding"]) > 0 