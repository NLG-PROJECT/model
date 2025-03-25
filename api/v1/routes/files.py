from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form, BackgroundTasks
from typing import List, Dict, Any
import logging
from ..schemas.file import FileResponse, UploadResponse
from ..services.file_service import FileService
from ..dependencies import get_file_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    save_to_drive: bool = Form(True),  # Default to True for backward compatibility
    file_service: FileService = Depends(get_file_service)
) -> UploadResponse:
    """Upload and process multiple files.
    
    Args:
        background: Background tasks handler
        files: List of files to upload
        save_to_drive: Whether to save files to Google Drive
        file_service: File service instance
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Process files
        results = await file_service.process_files(files, save_to_drive)
        
        # Log results
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count
        
        logger.info(f"File upload complete: {success_count} successful, {error_count} failed")
        
        return UploadResponse(
            message="Files processed successfully",
            processed_files=results
        )
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}", response_model=FileResponse)
async def get_file(doc_id: str, file_service: FileService = Depends(get_file_service)) -> FileResponse:
    """Get a single file by document ID."""
    try:
        file_info = await file_service.get_file(doc_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(**file_info)
    except Exception as e:
        logger.error(f"Error retrieving file {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[FileResponse])
async def list_files(file_service: FileService = Depends(get_file_service)) -> List[FileResponse]:
    """List all stored files."""
    try:
        files = await file_service.list_files()
        return [FileResponse(**file) for file in files]
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{doc_id}", response_model=Dict[str, str])
async def delete_file(
    doc_id: str,
    file_service: FileService = Depends(get_file_service)
) -> Dict[str, str]:
    """Delete a file by document ID."""
    try:
        success = await file_service.delete_file(doc_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {doc_id} not found"
            )
        return {"message": f"File {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        ) 