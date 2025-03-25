from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from .routes import base, files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Document Processing API",
        description="API for managing document processing and storage",
        version="1.0.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(base.router, prefix="/api/v1")
    app.include_router(files.router, prefix="/api/v1")
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting up Document Processing API")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down Document Processing API")
    
    return app 