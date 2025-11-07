"""Index management for Pinecone operations."""
import time
from typing import List, Optional

from pinecone import ServerlessSpec

from app.vectorstore.utils import (
    PROGRESS_UPDATE_INTERVAL,
    get_index_dimension,
    is_index_ready,
    log_message,
)


class IndexManager:
    """Manages Pinecone index creation and initialization."""

    def __init__(self, pinecone_client, index_name: str, embedding_dimension: int):
        """
        Initialize IndexManager.

        Args:
            pinecone_client: Pinecone client instance
            index_name: Name of the Pinecone index
            embedding_dimension: Dimension of embeddings
        """
        self.pinecone_client = pinecone_client
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension

    def _get_index_names(self) -> List[str]:
        """Get list of all index names from Pinecone."""
        try:
            indexes = self.pinecone_client.list_indexes()
            if hasattr(indexes, '__iter__'):
                return [idx.name if hasattr(idx, 'name') else str(idx) for idx in indexes]
            return []
        except Exception as e:
            raise ConnectionError(
                f"Failed to list Pinecone indexes. Please check your API key and internet connection. "
                f"Error: {str(e)}"
            )

    def _wait_for_index_ready(self, show_progress: bool = True):
        """Wait for index to become ready."""
        start_time = time.time()
        last_progress_time = 0

        while True:
            try:
                index_info = self.pinecone_client.describe_index(self.index_name)
                if is_index_ready(index_info):
                    return True
            except Exception:
                pass  # Continue waiting

            elapsed = time.time() - start_time
            if show_progress and int(elapsed) - last_progress_time >= PROGRESS_UPDATE_INTERVAL:
                print(f"   Still waiting... ({int(elapsed)}s elapsed)")
                last_progress_time = int(elapsed)

            time.sleep(2)

    def _create_index(self, show_progress: bool = True):
        """Create a new Pinecone index."""
        log_message(f"   Index '{self.index_name}' not found. Creating new index...", show_progress)

        self.pinecone_client.create_index(
            name=self.index_name,
            dimension=self.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        log_message("   Index created. Waiting for it to be ready...", show_progress)
        self._wait_for_index_ready(show_progress)
        log_message("   ✓ Index is ready", show_progress)

    def _verify_index_dimension(self, show_progress: bool = True):
        """Verify that existing index has the correct dimension."""
        try:
            index_info = self.pinecone_client.describe_index(self.index_name)
            existing_dimension = get_index_dimension(index_info)

            if existing_dimension and existing_dimension != self.embedding_dimension:
                raise ValueError(
                    f"Index dimension mismatch! "
                    f"Existing index '{self.index_name}' has dimension {existing_dimension}, "
                    f"but embeddings require dimension {self.embedding_dimension}. "
                    f"Please delete the existing index manually or create a new index with the correct dimension."
                )
            if existing_dimension:
                log_message(f"   ✓ Index dimension matches: {existing_dimension}", show_progress)
        except ValueError:
            raise
        except Exception as e:
            log_message(f"   ⚠️  Could not verify index dimension: {e}", show_progress)

    def initialize_index(self, show_progress: bool = True):
        """Initialize connection to Pinecone index, creating if needed."""
        log_message(f"   Checking if index '{self.index_name}' exists...", show_progress)

        index_names = self._get_index_names()

        if self.index_name not in index_names:
            self._create_index(show_progress)
        else:
            log_message(f"   ✓ Index '{self.index_name}' exists", show_progress)
            self._verify_index_dimension(show_progress)

