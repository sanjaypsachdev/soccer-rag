"""Query manager for filtering and retrieving documents by metadata."""
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from app.vectorstore.utils import DEFAULT_EMBEDDING_DIMENSION, MAX_QUERY_RESULTS


class QueryManager:
    """Manages query operations with metadata filtering."""

    def __init__(self, index, embedding_dimension: Optional[int] = None):
        """
        Initialize QueryManager.

        Args:
            index: Pinecone index instance
            embedding_dimension: Dimension of embeddings for dummy vector creation
        """
        self._index = index
        self._embedding_dimension = embedding_dimension

    def _create_dummy_vector(self) -> List[float]:
        """Create a dummy zero vector for queries that require a vector."""
        dimension = self._embedding_dimension or DEFAULT_EMBEDDING_DIMENSION
        return [0.0] * dimension

    def query_with_filter(self, filter_dict: Optional[Dict] = None) -> List:
        """Query index with optional metadata filter."""
        if not self._index:
            return []

        try:
            dummy_vector = self._create_dummy_vector()
            results = self._index.query(
                vector=dummy_vector,
                top_k=MAX_QUERY_RESULTS,
                include_metadata=True,
                filter=filter_dict
            )
            return results.matches
        except Exception as e:
            print(f"Error querying index: {e}")
            return []

    def delete_by_source_file(self, source_file: str) -> int:
        """
        Delete all documents from the vector store that came from a specific source file.

        Args:
            source_file: Path to the source file

        Returns:
            Number of documents deleted
        """
        if not self._index:
            raise ValueError("Pinecone index not accessible. Cannot delete documents.")

        try:
            documents = self.get_documents_by_source_file(source_file)
            count = len(documents)

            if count > 0:
                self._index.delete(filter={"source_file": source_file})

            return count
        except Exception as e:
            print(f"Error deleting documents from {source_file}: {e}")
            return 0

    def get_documents_by_source_file(self, source_file: str) -> List[Document]:
        """
        Get all documents from the vector store that came from a specific source file.

        Args:
            source_file: Path to the source file

        Returns:
            List of Document objects
        """
        matches = self.query_with_filter(filter_dict={"source_file": source_file})

        documents = []
        for match in matches:
            metadata = match.metadata or {}
            text = metadata.get('text', '') or metadata.get('page_content', '')
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def get_all_stored_source_files(self) -> Set[str]:
        """
        Get all unique source file paths currently stored in the vector store.

        Returns:
            Set of source file paths
        """
        if not self._index:
            return set()

        matches = self.query_with_filter()
        source_files = set()

        for match in matches:
            if match.metadata and 'source_file' in match.metadata:
                source_files.add(match.metadata['source_file'])

        return source_files

    def fetch_all_documents(self) -> List[Document]:
        """
        Fetch all documents from Pinecone.

        Returns:
            List of all Document objects in the index
        """
        matches = self.query_with_filter()
        documents = []
        for match in matches:
            metadata = match.metadata or {}
            text = metadata.get('text', '') or metadata.get('page_content', '')
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

