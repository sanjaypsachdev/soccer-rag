"""Embedder class for generating embeddings using OpenAI."""
from typing import List

from langchain_openai import OpenAIEmbeddings


class Embedder:
    """Generates embeddings using OpenAI model text-embedding-3-small."""

    def __init__(self):
        """Initialize the Embedder with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

