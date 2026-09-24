"""
retriever.py — Document Retrieval for RAG
==========================================
Provides retrieval with user/document isolation and configurable top-K.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None

from .embeddings import BGEEmbeddings, EmbeddingConfig
from .vectorstore import VectorStore, LocalVectorStore, VectorStoreConfig


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    similarity_threshold: Optional[float] = None  # Minimum similarity score
    include_metadata: bool = True
    include_scores: bool = True


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    documents: List[LangChainDocument]
    total_found: int
    config: RetrievalConfig
    user_id: str
    document_id: Optional[str] = None


class Retriever:
    """
    Retriever with user/document isolation.
    Uses dense retrieval with BGE embeddings.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: BGEEmbeddings,
        config: Optional[RetrievalConfig] = None
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        user_id: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            user_id: User ID for isolation (required)
            document_id: Optional document ID to restrict search
            top_k: Override default top_k
            filter: Additional metadata filters

        Returns:
            RetrievalResult with matched documents
        """
        if not user_id:
            raise ValueError("user_id is required for retrieval (isolation)")

        k = top_k or self.config.top_k

        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Search vector store with user isolation
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
            user_id=user_id,
            document_id=document_id
        )

        # Apply similarity threshold if configured
        if self.config.similarity_threshold is not None:
            results = [
                doc for doc in results
                if doc.metadata.get("similarity_score", 0) >= self.config.similarity_threshold
            ]

        return RetrievalResult(
            query=query,
            documents=results,
            total_found=len(results),
            config=self.config,
            user_id=user_id,
            document_id=document_id
        )

    def retrieve_multiple(
        self,
        queries: List[str],
        user_id: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve for multiple queries (e.g., question planning)."""
        return [
            self.retrieve(q, user_id, document_id, top_k)
            for q in queries
        ]


def create_retriever(
    user_id: str,
    document_id: Optional[str] = None,
    vector_store: Optional[VectorStore] = None,
    embeddings: Optional[BGEEmbeddings] = None,
    config: Optional[RetrievalConfig] = None,
    vector_store_config: Optional[VectorStoreConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> Retriever:
    """
    Factory function to create a retriever with all dependencies.
    """
    if vector_store is None:
        vector_store = LocalVectorStore(vector_store_config)

    if embeddings is None:
        embeddings = BGEEmbeddings(embedding_config)

    return Retriever(vector_store, embeddings, config)


def retrieve(
    query: str,
    user_id: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    vector_store: Optional[VectorStore] = None,
    embeddings: Optional[BGEEmbeddings] = None,
    vector_store_config: Optional[VectorStoreConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> RetrievalResult:
    """
    Convenience function for one-off retrieval.
    Creates temporary retriever if dependencies not provided.
    """
    retriever = create_retriever(
        user_id=user_id,
        document_id=document_id,
        vector_store=vector_store,
        embeddings=embeddings,
        config=RetrievalConfig(top_k=top_k),
        vector_store_config=vector_store_config,
        embedding_config=embedding_config,
    )
    return retriever.retrieve(query, user_id, document_id, top_k)


# Export
__all__ = [
    "RetrievalConfig",
    "RetrievalResult",
    "Retriever",
    "create_retriever",
    "retrieve",
]