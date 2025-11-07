"""Document operations for vectorstore (CRUD operations)."""
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from app.vectorstore.utils import DEFAULT_BATCH_SIZE


class DocumentOperations:
    """Handles document CRUD operations for the vectorstore."""

    def __init__(self, vectorstore: PineconeVectorStore):
        """
        Initialize DocumentOperations.

        Args:
            vectorstore: PineconeVectorStore instance
        """
        self.vectorstore = vectorstore

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects (can include metadata)
            batch_size: Number of documents to process in each batch
        """
        if not documents:
            return

        for batch_idx in range(0, len(documents), batch_size):
            batch = documents[batch_idx:batch_idx + batch_size]
            self.vectorstore.add_documents(batch)

    def add_texts(self, texts: List[str]):
        """Add texts directly to the vector store."""
        self.vectorstore.add_texts(texts)

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """Perform similarity search in the vector store."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with scores."""
        return self.vectorstore.similarity_search_with_score(query, k=k)

