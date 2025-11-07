"""Hybrid retriever combining semantic and BM25 search."""
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_community.retrievers import BM25Retriever


class HybridRetriever(BaseRetriever):
    """LCEL-based hybrid retriever combining semantic and BM25."""

    def __init__(self, chain):
        super().__init__()
        self.chain = chain

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.chain.invoke(query)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return await self.chain.ainvoke(query)


def create_hybrid_retriever(
    semantic_retriever: BaseRetriever,
    bm25_retriever: BM25Retriever,
    k: int,
    weights: Optional[List[float]] = None
) -> HybridRetriever:
    """
    Create a hybrid retriever combining semantic and BM25 search using LCEL.

    Args:
        semantic_retriever: Semantic retriever from vectorstore
        bm25_retriever: BM25 retriever instance
        k: Number of results to retrieve
        weights: Weights for [semantic, bm25] (default: [0.5, 0.5])

    Returns:
        HybridRetriever instance
    """
    if weights is None:
        weights = [0.5, 0.5]

    semantic_weight, bm25_weight = weights

    # Set k for BM25 retriever
    if hasattr(bm25_retriever, 'k'):
        bm25_retriever.k = k

    def combine_results(inputs: Dict[str, List[Document]]) -> List[Document]:
        """Combine results from both retrievers and rerank using weighted scores."""
        semantic_docs = inputs.get("semantic", [])
        bm25_docs = inputs.get("bm25", [])

        doc_scores = {}

        # Score semantic results (inverse rank * weight)
        for rank, doc in enumerate(semantic_docs, start=1):
            doc_key = doc.page_content[:200]  # Use content as key for deduplication
            score = semantic_weight * (1.0 / rank)

            if doc_key not in doc_scores:
                doc_scores[doc_key] = (doc, score)
            else:
                existing_doc, existing_score = doc_scores[doc_key]
                doc_scores[doc_key] = (existing_doc, existing_score + score)

        # Score BM25 results (inverse rank * weight)
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_key = doc.page_content[:200]
            score = bm25_weight * (1.0 / rank)

            if doc_key not in doc_scores:
                doc_scores[doc_key] = (doc, score)
            else:
                existing_doc, existing_score = doc_scores[doc_key]
                doc_scores[doc_key] = (existing_doc, existing_score + score)

        # Sort by combined score and return top k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:k]]

    # Create LCEL chain: parallel retrieval -> combine & rerank
    hybrid_chain = (
        RunnableParallel({
            "semantic": semantic_retriever,
            "bm25": bm25_retriever,
        })
        | RunnableLambda(combine_results)
    )

    return HybridRetriever(hybrid_chain)


def build_bm25_retriever(documents: List[Document]) -> Optional[BM25Retriever]:
    """
    Build BM25 retriever from documents.

    Args:
        documents: List of documents to index

    Returns:
        BM25Retriever instance or None if no documents
    """
    if not documents:
        return None

    try:
        return BM25Retriever.from_documents(documents)
    except (AttributeError, TypeError):
        # Fallback to from_texts if from_documents is not available
        texts = [doc.page_content for doc in documents]
        return BM25Retriever.from_texts(texts)

