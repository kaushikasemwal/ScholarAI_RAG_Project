"""
embeddings.py — BGE Embeddings for RAG
=======================================
Reuses the existing BAAI/bge-base-en-v1.5 model from the model manager.
Provides LangChain-compatible embeddings interface.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
except ImportError:
    LangChainEmbeddings = object

from ..models import get_sbert
from ..config import get_settings


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "BAAI/bge-base-en-v1.5"
    expected_dimension: int = 768
    normalize_embeddings: bool = True
    batch_size: int = 32
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""


class BGEEmbeddings(LangChainEmbeddings):
    """
    LangChain-compatible wrapper for BGE embeddings.
    Reuses the shared SBERT model from models/__init__.py.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    @property
    def model(self):
        """Lazy-load the shared SBERT model."""
        if self._model is None:
            self._model = get_sbert()
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        Uses document_prefix for retrieval-optimized embeddings.
        """
        if not texts:
            return []

        # Prepend document prefix if configured
        if self.config.document_prefix:
            texts = [self.config.document_prefix + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        Uses query_prefix for retrieval-optimized embeddings (BGE recommendation).
        """
        if self.config.query_prefix:
            text = self.config.query_prefix + text

        embedding = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embedding[0].tolist()

    def embed_chunks(self, chunks: List["LangChainDocument"]) -> List[List[float]]:
        """
        Embed a list of LangChain Documents.
        Extracts page_content from each document.
        """
        texts = [chunk.page_content for chunk in chunks]
        return self.embed_documents(texts)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.expected_dimension

    def verify_dimension(self) -> bool:
        """Verify the model produces expected dimension embeddings."""
        test_emb = self.embed_query("test")
        return len(test_emb) == self.config.expected_dimension


def get_embeddings(config: Optional[EmbeddingConfig] = None) -> BGEEmbeddings:
    """
    Get the BGE embeddings instance.
    Uses the shared model manager to avoid duplicate model loading.
    """
    return BGEEmbeddings(config)


def create_embedding_config_from_settings() -> EmbeddingConfig:
    """Create EmbeddingConfig from application settings."""
    settings = get_settings()
    return EmbeddingConfig(
        model_name=settings.SBERT_MODEL,
        expected_dimension=768,  # BGE-base is 768-dim
        normalize_embeddings=True,
        batch_size=32,
    )


# Export
__all__ = [
    "EmbeddingConfig",
    "BGEEmbeddings",
    "get_embeddings",
    "create_embedding_config_from_settings",
]