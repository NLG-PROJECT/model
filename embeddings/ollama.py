from typing import List, Dict, Any
import numpy as np
from langchain_ollama import OllamaEmbeddings
from .base import EmbeddingService
import logging
import time
import asyncio

logger = logging.getLogger(__name__)

class OllamaEmbeddingService(EmbeddingService):
    """Implementation of embedding service using Ollama with optimized caching and request management."""
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 8192,
        batch_size: int = 20,
        cache_size: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize Ollama client
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        
        # Initialize caches
        self._memory_cache = {}  # Fast in-memory cache
        self._model_info = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._request_count = 0
        self._error_count = 0
        
        # Performance metrics
        self._total_processing_time = 0
        self._batch_processing_times = []
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            # Check cache first
            if text in self._memory_cache:
                return self._memory_cache[text]
            
            if self.is_text_too_long(text):
                text = self.truncate_text(text)
            
            # Use embed_documents with a single text for better error handling
            embeddings = await self.embeddings.aembed_documents([text])
            embedding = np.array(embeddings[0], dtype=np.float32)
            
            # Cache the result
            self._memory_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with optimized caching and batching."""
        start_time = time.time()
        self._request_count += 1
        
        try:
            # Check memory cache first
            uncached_texts = []
            cached_embeddings = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if text in self._memory_cache:
                    cached_embeddings.append(self._memory_cache[text])
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    text_indices.append(i)
                    self._cache_misses += 1
            
            if not uncached_texts:
                self._total_processing_time += time.time() - start_time
                return cached_embeddings
            
            # Process uncached texts in optimized batches
            new_embeddings = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i + self.batch_size]
                batch_start_time = time.time()
                
                # Retry logic for batch processing
                for attempt in range(self.max_retries):
                    try:
                        batch_embeddings = await self.embeddings.aembed_documents(batch)
                        batch_embeddings = [np.array(embedding, dtype=np.float32) 
                                          for embedding in batch_embeddings]
                        new_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self._error_count += 1
                            logger.error(f"Failed to generate embeddings after {self.max_retries} attempts: {e}")
                            raise
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                
                batch_time = time.time() - batch_start_time
                self._batch_processing_times.append(batch_time)
            
            # Update memory cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                if len(self._memory_cache) >= self.cache_size:
                    # Remove oldest entry (FIFO)
                    self._memory_cache.pop(next(iter(self._memory_cache)))
                self._memory_cache[text] = embedding
            
            # Combine cached and new embeddings in the original order
            result = [None] * len(texts)
            cache_idx = 0
            new_idx = 0
            
            for i, text in enumerate(texts):
                if text in self._memory_cache:
                    result[i] = self._memory_cache[text]
                    cache_idx += 1
                else:
                    result[i] = new_embeddings[new_idx]
                    new_idx += 1
            
            # Log performance metrics
            total_time = time.time() - start_time
            self._total_processing_time += total_time
            
            if self._request_count % 100 == 0:  # Log every 100 requests
                avg_time = self._total_processing_time / self._request_count
                avg_batch_time = sum(self._batch_processing_times) / len(self._batch_processing_times)
                hit_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100
                
                logger.info(f"Performance metrics:")
                logger.info(f"- Average processing time: {avg_time:.2f}s")
                logger.info(f"- Average batch time: {avg_batch_time:.2f}s")
                logger.info(f"- Cache hit rate: {hit_rate:.2f}%")
                logger.info(f"- Error rate: {(self._error_count / self._request_count) * 100:.2f}%")
            
            return result
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model and performance metrics."""
        if self._model_info is None:
            try:
                self._model_info = {
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "max_tokens": self.max_tokens,
                    "embedding_dimension": self.get_embedding_dimension(),
                    "cache_size": self.cache_size,
                    "batch_size": self.batch_size,
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "request_count": self._request_count,
                    "error_count": self._error_count,
                    "average_processing_time": self._total_processing_time / max(1, self._request_count),
                    "average_batch_time": sum(self._batch_processing_times) / max(1, len(self._batch_processing_times))
                }
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                self._model_info = {}
        return self._model_info
    
    async def validate_connection(self) -> bool:
        """Validate connection to the embedding service."""
        try:
            # Try to generate a test embedding
            test_text = "Test connection"
            await self.generate_embedding(test_text)
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        try:
            # Generate a test embedding to get dimension
            test_text = "Test dimension"
            embedding = await self.generate_embedding(test_text)
            return len(embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 768  # Default dimension for nomic-embed-text
    
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens the model can handle."""
        return self.max_tokens
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # Simple estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def is_text_too_long(self, text: str) -> bool:
        """Check if text is too long for the model."""
        return self.get_token_count(text) > self.max_tokens
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within model limits."""
        max_chars = self.max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text 