from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check the health of the API and its dependencies."""
    try:
        return {
            "status": "healthy",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        ) 