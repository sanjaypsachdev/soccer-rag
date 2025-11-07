"""Utility functions and constants for vectorstore operations."""
from typing import Optional

# Constants
DEFAULT_EMBEDDING_DIMENSION = 1536
PROGRESS_UPDATE_INTERVAL = 5
DEFAULT_BATCH_SIZE = 100
MAX_QUERY_RESULTS = 10000


def get_embedding_dimension(embeddings) -> int:
    """Get the embedding dimension from the embeddings model."""
    try:
        test_embedding = embeddings.embed_query("test")
        return len(test_embedding)
    except Exception:
        return DEFAULT_EMBEDDING_DIMENSION


def get_index_dimension(index_info) -> Optional[int]:
    """Extract dimension from index info using multiple fallback methods."""
    if hasattr(index_info, 'dimension'):
        return index_info.dimension
    if hasattr(index_info, 'spec') and hasattr(index_info.spec, 'dimension'):
        return index_info.spec.dimension
    if isinstance(index_info, dict):
        return index_info.get('dimension')
    return None


def is_index_ready(index_info) -> bool:
    """Check if index is ready using multiple fallback methods."""
    if hasattr(index_info, 'status'):
        status = index_info.status
        if isinstance(status, dict):
            if status.get('ready', False):
                return True
        elif isinstance(status, str) and status.lower() in ['ready', 'readyforquery']:
            return True
    
    if hasattr(index_info, 'ready'):
        return bool(index_info.ready)
    
    return False


def log_message(message: str, show_progress: bool = True):
    """Log a message if progress is enabled."""
    if show_progress:
        print(message)

