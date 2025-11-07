"""Main Vectorstore class for managing Pinecone vector database operations."""
import os
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.vectorstore.bm25_manager import BM25Manager
from app.vectorstore.document_operations import DocumentOperations
from app.vectorstore.hybrid_retriever import create_hybrid_retriever
from app.vectorstore.index_manager import IndexManager
from app.vectorstore.query_manager import QueryManager
from app.vectorstore.sync_manager import SyncManager
from app.vectorstore.utils import get_embedding_dimension, log_message


class Vectorstore:
    """Manages Pinecone vector database operations."""

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        enable_hybrid_search: bool = False,
    ):
        """
        Initialize the Vectorstore.

        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            enable_hybrid_search: Whether to enable hybrid search (semantic + BM25)
        """
        self.index_name = index_name
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter."
            )

        try:
            self.pinecone_client = Pinecone(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Pinecone client: {e}")

        self.vectorstore: Optional[PineconeVectorStore] = None
        self._index = None
        self._embedding_dimension: Optional[int] = None
        self.enable_hybrid_search = enable_hybrid_search

        # Managers (initialized after vectorstore is set up)
        self._document_ops: Optional[DocumentOperations] = None
        self._query_manager: Optional[QueryManager] = None
        self._bm25_manager: Optional[BM25Manager] = None
        self._index_manager: Optional[IndexManager] = None
        self._sync_manager: Optional[SyncManager] = None

    def _ensure_initialized(self):
        """Ensure vectorstore is initialized."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call initialize_vectorstore first.")

    def _setup_vectorstore(self, embeddings):
        """Set up the LangChain vectorstore and underlying Pinecone index."""
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
        )

        # Store reference to underlying Pinecone index
        try:
            self._index = self.pinecone_client.Index(self.index_name)
        except Exception:
            try:
                self._index = self.vectorstore._index
            except Exception:
                pass

        # Initialize managers after vectorstore is set up
        self._document_ops = DocumentOperations(self.vectorstore)
        self._query_manager = QueryManager(self._index, self._embedding_dimension)
        self._bm25_manager = BM25Manager(self._query_manager)

    def initialize_vectorstore(self, embeddings, show_progress: bool = True):
        """
        Initialize the LangChain Pinecone vector store.
        Creates the index if it doesn't exist.

        Args:
            embeddings: Embeddings instance (e.g., OpenAIEmbeddings)
            show_progress: Whether to show progress messages

        Raises:
            ValueError: If the existing index has a dimension mismatch
            ConnectionError: If connection to Pinecone fails
        """
        log_message("   Connecting to Pinecone...", show_progress)

        self._embedding_dimension = get_embedding_dimension(embeddings)
        log_message(f"   Embedding dimension: {self._embedding_dimension}", show_progress)

        try:
            self._index_manager = IndexManager(
                self.pinecone_client, self.index_name, self._embedding_dimension
            )
            self._index_manager.initialize_index(show_progress)
            log_message("   Initializing vector store connection...", show_progress)
            self._setup_vectorstore(embeddings)
            log_message("   ✓ Vector store initialized successfully", show_progress)
        except ConnectionError:
            raise
        except ValueError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                raise ConnectionError(
                    f"Failed to connect to Pinecone. Please check your internet connection and API key. "
                    f"Error: {error_msg}"
                )
            raise ValueError(f"Failed to initialize vector store: {error_msg}")

    # Document operations - delegate to DocumentOperations
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ):
        """Add documents to the vector store."""
        self._ensure_initialized()
        self._document_ops.add_documents(documents, batch_size)

    def add_texts(self, texts: List[str]):
        """Add texts directly to the vector store."""
        self._ensure_initialized()
        self._document_ops.add_texts(texts)

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """Perform similarity search in the vector store."""
        self._ensure_initialized()
        return self._document_ops.similarity_search(query, k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with scores."""
        self._ensure_initialized()
        return self._document_ops.similarity_search_with_score(query, k)

    # Retriever operations
    def as_retriever(self, k: int = 4, search_kwargs: Optional[Dict] = None):
        """Get a retriever from the vectorstore."""
        if self.enable_hybrid_search:
            return self.as_hybrid_retriever(k=k, search_kwargs=search_kwargs)

        self._ensure_initialized()
        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs.setdefault("k", k)
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def as_hybrid_retriever(
        self,
        k: int = 4,
        search_kwargs: Optional[Dict] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Get a hybrid retriever combining semantic and BM25 search.

        Args:
            k: Number of results to retrieve
            search_kwargs: Additional search kwargs for semantic retriever
            weights: Weights for [semantic, bm25] (default: [0.5, 0.5])

        Returns:
            BaseRetriever instance that combines both retrievers
        """
        self._ensure_initialized()

        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs.setdefault("k", k)

        semantic_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Build BM25 retriever if not already built
        if self._bm25_manager.retriever is None:
            self._bm25_manager.build_retriever()

        bm25_retriever = self._bm25_manager.retriever
        if bm25_retriever is None:
            return semantic_retriever

        return create_hybrid_retriever(semantic_retriever, bm25_retriever, k, weights)

    # Query operations - delegate to QueryManager
    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all documents from the vector store that came from a specific source file."""
        self._ensure_initialized()
        return self._query_manager.delete_by_source_file(source_file)

    def get_documents_by_source_file(self, source_file: str) -> List[Document]:
        """Get all documents from the vector store that came from a specific source file."""
        self._ensure_initialized()
        return self._query_manager.get_documents_by_source_file(source_file)

    def get_all_stored_source_files(self) -> Set[str]:
        """Get all unique source file paths currently stored in the vector store."""
        if not self._index or not self._query_manager:
            return set()
        return self._query_manager.get_all_stored_source_files()

    # Sync operations - delegate to SyncManager
    def sync_with_datasets(self, document_loader, datasets_dir, show_progress: bool = True) -> Dict[str, int]:
        """
        Sync the vector store with the datasets directory.

        Args:
            document_loader: DocumentLoader instance
            datasets_dir: Path to the datasets directory
            show_progress: Whether to show progress indicators

        Returns:
            Dictionary with counts of deleted, updated, and added documents
        """
        self._ensure_initialized()

        if self._sync_manager is None:
            self._sync_manager = SyncManager(self)

        stats = self._sync_manager.sync_with_datasets(document_loader, datasets_dir, show_progress)

        # Rebuild BM25 index if hybrid search is enabled
        if self.enable_hybrid_search:
            self._bm25_manager.build_retriever()

        return stats
