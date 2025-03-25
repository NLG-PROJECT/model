from typing import List, Dict, Any
import numpy as np
from langchain_ollama import OllamaEmbeddings
from .base import EmbeddingService
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddingService(EmbeddingService):
    """Implementation of embedding service using Ollama."""
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 8192
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        self._model_info = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            if self.is_text_too_long(text):
                text = self.truncate_text(text)
            
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        try:
            # Truncate texts if needed
            texts = [self.truncate_text(text) if self.is_text_too_long(text) else text 
                    for text in texts]
            
            embeddings = self.embeddings.embed_documents(texts)
            return [np.array(embedding, dtype=np.float32) for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        if self._model_info is None:
            try:
                # Get model info from Ollama
                self._model_info = {
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "max_tokens": self.max_tokens,
                    "embedding_dimension": self.get_embedding_dimension()
                }
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                self._model_info = {}
        return self._model_info
    
    def validate_connection(self) -> bool:
        """Validate connection to the embedding service."""
        try:
            # Try to generate a test embedding
            test_text = "Test connection"
            self.generate_embedding(test_text)
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        try:
            # Generate a test embedding to get dimension
            test_text = "Test dimension"
            embedding = self.generate_embedding(test_text)
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