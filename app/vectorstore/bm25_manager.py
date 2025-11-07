"""BM25 retriever management for hybrid search."""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from app.vectorstore.hybrid_retriever import build_bm25_retriever
from app.vectorstore.query_manager import QueryManager


class BM25Manager:
    """Manages BM25 retriever lifecycle and document caching."""

    def __init__(self, query_manager: QueryManager):
        """
        Initialize BM25Manager.

        Args:
            query_manager: QueryManager instance for fetching documents
        """
        self.query_manager = query_manager
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._all_documents: List[Document] = []

    def build_retriever(self, documents: Optional[List[Document]] = None) -> Optional[BM25Retriever]:
        """
        Build or rebuild BM25 retriever from documents.

        Args:
            documents: Optional list of documents to use. If None, fetches from query manager.

        Returns:
            BM25Retriever instance or None if no documents available
        """
        if documents is not None:
            self._all_documents = documents
        elif not self._all_documents:
            self._all_documents = self.query_manager.fetch_all_documents()

        self._bm25_retriever = build_bm25_retriever(self._all_documents)
        return self._bm25_retriever

    @property
    def retriever(self) -> Optional[BM25Retriever]:
        """Get the current BM25 retriever instance."""
        return self._bm25_retriever

    def clear_cache(self):
        """Clear the cached documents."""
        self._all_documents = []
        self._bm25_retriever = None

