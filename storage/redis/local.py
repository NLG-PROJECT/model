import redis
import logging
import os
from .base import BaseRedisStorage
from config.storage import RedisConfig
import json
from typing import Any, Optional

logger = logging.getLogger(__name__)

class LocalRedisStorage(BaseRedisStorage):
    """Local Redis storage implementation."""
    
    def __init__(
        self,
        config: RedisConfig,
        prefix: str = "docstore"
    ):
        """Initialize local Redis storage.
        
        Args:
            config: Redis configuration
            prefix: Prefix for all Redis keys
        """
        super().__init__(prefix=prefix)
        self.config = config
        self.client = self._create_redis_client()
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with appropriate configuration."""
        try:
            # Build Redis URL without password
            redis_url = f"redis://{self.config.host}:{self.config.port}/{self.config.db}"
            logger.info(f"Creating Redis client with URL: {redis_url}")
            
            # Create client with URL and additional parameters
            client = redis.from_url(
                redis_url,
                decode_responses=False,  # Need binary mode for vector storage
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            logger.info("Testing Redis connection...")
            if client.ping():
                logger.info("Successfully connected to local Redis")
            else:
                logger.warning("Redis ping failed")
            
            # Test if we can list keys
            logger.info("Testing Redis key listing...")
            keys = client.keys("*")
            logger.info(f"Found {len(keys)} keys in Redis")
            
            return client
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to local Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating local Redis client: {e}")
            raise 