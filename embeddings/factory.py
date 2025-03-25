from typing import Dict, Type
from .base import EmbeddingService
from .ollama import OllamaEmbeddingService
import os
import logging

logger = logging.getLogger(__name__)

class EmbeddingServiceFactory:
    """Factory for creating embedding service instances."""
    
    _services: Dict[str, Type[EmbeddingService]] = {
        "ollama": OllamaEmbeddingService,
        # Add more services here as they are implemented
    }
    
    @classmethod
    def register_service(cls, name: str, service_class: Type[EmbeddingService]) -> None:
        """Register a new embedding service."""
        cls._services[name] = service_class
    
    @classmethod
    def get_service(cls, name: str = None, **kwargs) -> EmbeddingService:
        """Get an embedding service instance."""
        # If no name provided, try to get from environment
        if name is None:
            name = os.getenv("EMBEDDING_PROVIDER", "ollama")
        
        if name not in cls._services:
            raise ValueError(f"Unknown embedding service: {name}")
        
        try:
            service_class = cls._services[name]
            return service_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating embedding service {name}: {e}")
            raise
    
    @classmethod
    def get_available_services(cls) -> list:
        """Get list of available embedding services."""
        return list(cls._services.keys())
    
    @classmethod
    def validate_service(cls, name: str) -> bool:
        """Check if a service name is valid."""
        return name in cls._services 