from typing import Dict, Type
from .base import StorageInterface
from .redis import LocalRedisStorage, UpstashRedisStorage
import os
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class StorageFactory:
    """Factory for creating storage service instances."""
    
    _services: Dict[str, Type[StorageInterface]] = {
        "redis": LocalRedisStorage,  # Default to local Redis
        "upstash": UpstashRedisStorage,
    }
    
    @classmethod
    def register_service(cls, name: str, service_class: Type[StorageInterface]) -> None:
        """Register a new storage service."""
        cls._services[name] = service_class
    
    @classmethod
    def get_service(cls, name: str = None, **kwargs) -> StorageInterface:
        """Get a storage service instance."""
        # If no name provided, try to determine from environment
        if name is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            parsed_url = urlparse(redis_url)
            if "upstash.io" in parsed_url.netloc:
                name = "upstash"
            else:
                name = "redis"
        
        if name not in cls._services:
            raise ValueError(f"Unknown storage service: {name}")
        
        try:
            service_class = cls._services[name]
            return service_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating storage service {name}: {e}")
            raise
    
    @classmethod
    def get_available_services(cls) -> list:
        """Get list of available storage services."""
        return list(cls._services.keys())
    
    @classmethod
    def validate_service(cls, name: str) -> bool:
        """Check if a service name is valid."""
        return name in cls._services 