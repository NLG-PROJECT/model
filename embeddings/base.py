from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to the embedding service."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens the model can handle."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        pass
    
    @abstractmethod
    def is_text_too_long(self, text: str) -> bool:
        """Check if text is too long for the model."""
        pass
    
    @abstractmethod
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within model limits."""
        pass 