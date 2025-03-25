# Document Processing System

A modular document processing system that supports various document types, with features for chunking, embedding, and storage.

## Features

- Support for multiple document types (PDF, DOCX, Excel, etc.)
- Configurable text chunking with overlap
- Flexible embedding service integration (currently supports Ollama)
- Redis-based storage with support for both local and Upstash
- Comprehensive test coverage
- Environment-based configuration

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The system can be configured through environment variables:

### Redis Configuration

- `REDIS_URL`: Redis connection URL (default: "redis://localhost:6379")
- `REDIS_PREFIX`: Prefix for Redis keys (default: "docstore")

### Embedding Configuration

- `EMBEDDING_PROVIDER`: Embedding service provider (default: "ollama")
- `EMBEDDING_MODEL`: Model to use for embeddings (default: "llama2")
- `EMBEDDING_DIMENSION`: Embedding dimension (default: 4096)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: 32)

### Chunking Configuration

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `MIN_CHUNK_SIZE`: Minimum chunk size (default: 100)

### Debug Mode

- `DEBUG`: Enable debug mode (default: false)

## Usage

### Basic Usage

```python
from storage.factory import StorageFactory
from embeddings.factory import EmbeddingServiceFactory
from config import config

# Initialize services
storage = StorageFactory.get_service()
embedding_service = EmbeddingServiceFactory.get_service()

# Process a document
# ... (document processing code)
```

### Running Tests

```bash
pytest tests/
```

## Project Structure

```
.
├── config.py              # Configuration management
├── requirements.txt       # Project dependencies
├── storage/              # Storage implementations
│   ├── base.py          # Storage interface
│   ├── factory.py       # Storage factory
│   └── redis.py         # Redis implementation
├── embeddings/          # Embedding services
│   ├── base.py         # Embedding interface
│   ├── factory.py      # Embedding factory
│   └── ollama.py       # Ollama implementation
└── tests/              # Test files
    ├── test_config.py
    └── test_storage.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

#start redis
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

#run environment
uvicorn main:app --reload

#requirements
ensure you have ollama running
