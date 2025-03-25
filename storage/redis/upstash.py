import redis
import logging
import os
from .base import BaseRedisStorage

logger = logging.getLogger(__name__)

class UpstashRedisStorage(BaseRedisStorage):
    """Upstash Redis storage implementation."""
    
    def __init__(
        self,
        redis_url: str = None,
        prefix: str = "docstore",
        username: str = None,
        password: str = None
    ):
        """Initialize Upstash Redis storage.
        
        Args:
            redis_url: Upstash Redis connection URL. If None, will try to get from environment.
            prefix: Prefix for all Redis keys.
            username: Upstash username. If None, will try to get from environment.
            password: Upstash password. If None, will try to get from environment.
        """
        super().__init__(prefix=prefix)
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.username = username or os.getenv("REDIS_USERNAME")
        self.password = password or os.getenv("REDIS_PASSWORD")
        
        # Validate required parameters
        if not self.redis_url:
            raise ValueError("REDIS_URL is required for Upstash")
        if not self.username:
            raise ValueError("REDIS_USERNAME is required for Upstash")
        if not self.password:
            raise ValueError("REDIS_PASSWORD is required for Upstash")
        
        self.client = self._create_redis_client()
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with appropriate configuration."""
        try:
            # Basic connection parameters
            connection_params = {
                "url": self.redis_url,
                "decode_responses": False,  # Need binary mode for vector storage
                "ssl": True,  # Upstash requires SSL
                "username": self.username,
                "password": self.password
            }
            
            # Create client
            client = redis.from_url(**connection_params)
            
            # Test connection
            client.ping()
            logger.info("Successfully connected to Upstash Redis")
            
            return client
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Upstash Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating Upstash Redis client: {e}")
            raise 