"""
vectorstore.py — Vector Store Abstraction for RAG
================================================
Provides a pluggable vector store interface with local Chroma implementation.
Designed for future migration to Cloudflare Vectorize.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import os

try:
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.vectorstores import VectorStore as LangChainVectorStore
except ImportError:
    LangChainDocument = None
    LangChainVectorStore = object

log = logging.getLogger(__name__)


# Collection metadata keys for embedding dimension tracking
EMBEDDING_DIMENSION_KEY = "embedding_dimension"
EMBEDDING_MODEL_KEY = "embedding_model"


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "scholarai_documents"
    distance_metric: str = "cosine"  # cosine, l2, ip
    # Embedding dimension validation
    expected_embedding_dimension: int = 768
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    # Future: Cloudflare Vectorize config
    vectorize_index_name: Optional[str] = None
    vectorize_account_id: Optional[str] = None
    vectorize_api_token: Optional[str] = None


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    All implementations must provide these methods.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[LangChainDocument],
        embeddings: List[List[float]],
        user_id: str,
        document_id: str
    ) -> List[str]:
        """
        Add documents with their embeddings to the store.
        Returns list of document IDs.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[LangChainDocument]:
        """
        Search for similar documents.
        Supports metadata filtering for user/document isolation.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete all chunks for a document."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents (use with caution)."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass


class LocalVectorStore(VectorStore):
    """
    Local Chroma-based vector store implementation.
    Provides document/user isolation via metadata filtering.
    Includes embedding dimension validation to prevent mismatch.
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embedding_function=None
    ):
        self.config = config or VectorStoreConfig()
        self.embedding_function = embedding_function
        self._client = None
        self._collection = None
        self._initialize()
        self._validate_collection_dimension()

    def _initialize(self):
        """Initialize Chroma client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        # Ensure persist directory exists
        os.makedirs(self.config.persist_directory, exist_ok=True)

        # Create Chroma client with persistence
        self._client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection with embedding metadata
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "hnsw:space": self.config.distance_metric,
                EMBEDDING_DIMENSION_KEY: self.config.expected_embedding_dimension,
                EMBEDDING_MODEL_KEY: self.config.embedding_model_name,
            }
        )
        log.info(f"Initialized Chroma collection: {self.config.collection_name} "
                 f"(expected dim: {self.config.expected_embedding_dimension}, "
                 f"model: {self.config.embedding_model_name})")

    def _validate_collection_dimension(self):
        """Validate that the collection's embedding dimension matches configuration."""
        try:
            metadata = self._collection.metadata or {}
            stored_dim = metadata.get(EMBEDDING_DIMENSION_KEY)
            stored_model = metadata.get(EMBEDDING_MODEL_KEY)
            
            if stored_dim is not None and stored_dim != self.config.expected_embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch detected!\n"
                    f"  Configured embedding model: {self.config.embedding_model_name}\n"
                    f"  Configured dimension: {self.config.expected_embedding_dimension}\n"
                    f"  Collection's stored dimension: {stored_dim}\n"
                    f"  Collection's stored model: {stored_model}\n"
                    f"\n"
                    f"Resolution options:\n"
                    f"  1. Set RAG_EMBEDDING_TIER to match the collection's dimension\n"
                    f"  2. Use a different collection name (RAG_COLLECTION_NAME)\n"
                    f"  3. Clear the existing ChromaDB data (delete {self.config.persist_directory})\n"
                    f"  4. Set RAG_EMBEDDING_MODEL to a model with dimension {stored_dim}"
                )
            elif stored_dim is None:
                # Collection exists but has no dimension metadata (legacy)
                # Log warning but don't fail - will validate on first insert
                log.warning(
                    f"Collection '{self.config.collection_name}' has no embedding dimension metadata. "
                    f"This may indicate legacy data. First insertion will validate dimension."
                )
        except ValueError:
            raise
        except Exception as e:
            log.warning(f"Could not validate collection dimension: {e}")

    def add_documents(
        self,
        documents: List[LangChainDocument],
        embeddings: List[List[float]],
        user_id: str,
        document_id: str
    ) -> List[str]:
        """Add documents with embeddings to Chroma."""
        if not documents:
            return []

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        # Validate embedding dimension on first insert
        if embeddings:
            actual_dim = len(embeddings[0])
            if actual_dim != self.config.expected_embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch on insert!\n"
                    f"  Expected dimension: {self.config.expected_embedding_dimension} "
                    f"(from model: {self.config.embedding_model_name})\n"
                    f"  Actual embedding dimension: {actual_dim}\n"
                    f"\n"
                    f"Resolution options:\n"
                    f"  1. Ensure RAG_EMBEDDING_TIER matches the configured embedding model\n"
                    f"  2. Set RAG_EMBEDDING_MODEL to a model with dimension {actual_dim}\n"
                    f"  3. Use a different collection name (RAG_COLLECTION_NAME)\n"
                    f"  4. Clear the existing ChromaDB data"
                )

        # Prepare data for Chroma
        ids = []
        texts = []
        metadatas = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            # Generate unique ID for this chunk
            chunk_id = doc.metadata.get("chunk_id", f"{document_id}_chunk_{i}")
            ids.append(chunk_id)
            texts.append(doc.page_content)

            # Ensure user_id and document_id are in metadata for filtering
            metadata = {
                **doc.metadata,
                "user_id": user_id,
                "document_id": document_id,
            }
            metadatas.append(metadata)

        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        log.info(f"Added {len(ids)} chunks to vector store for document {document_id} (user: {user_id})")
        return ids

    def _build_where_filter(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB-compatible where filter.
        
        ChromaDB 1.5.9 requires explicit $and operator for multi-condition filters.
        Single conditions can be passed directly.
        
        If the provided filter already contains operators ($and, $or), it is used as-is
        with user_id/document_id appended to the $and array, or wrapped if needed.
        """
        # Check if the provided filter already contains operators
        has_operators = False
        if filter and any(k.startswith('$') for k in filter.keys()):
            has_operators = True
        
        # Build the base conditions
        base_conditions = {}
        if user_id:
            base_conditions["user_id"] = user_id
        if document_id:
            base_conditions["document_id"] = document_id
        
        # If no filter provided, just use base conditions
        if not filter:
            if not base_conditions:
                return None
            if len(base_conditions) == 1:
                return base_conditions
            return {"$and": [{k: v} for k, v in base_conditions.items()]}
        
        # Filter provided - check if it has operators
        if has_operators:
            # Filter has operators ($and, $or, etc.) - we need to combine carefully
            # If it's $and, we can append our conditions
            if '$and' in filter:
                # Append base conditions to existing $and array
                combined = filter.copy()
                combined['$and'] = filter['$and'] + [{k: v} for k, v in base_conditions.items()]
                return combined
            else:
                # Other operators ($or, etc.) - wrap everything in $and
                combined = {}
                if base_conditions:
                    combined['$and'] = [{k: v} for k, v in base_conditions.items()]
                # Add the operator filter as a condition
                for k, v in filter.items():
                    if combined.get('$and'):
                        combined['$and'].append({k: v})
                    else:
                        combined[k] = v
                if len(combined) == 1 and '$and' not in combined:
                    return list(combined.values())[0]
                return combined
        else:
            # Filter has no operators - merge with base conditions
            where_filter = {**base_conditions, **filter}
            if not where_filter:
                return None
            if len(where_filter) == 1:
                return where_filter
            return {"$and": [{k: v} for k, v in where_filter.items()]}

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[LangChainDocument]:
        """Search for similar documents with optional metadata filtering."""
        where_filter = self._build_where_filter(user_id, document_id, filter)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to LangChain Documents
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, text, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (for cosine: 1 - distance)
                if self.config.distance_metric == "cosine":
                    score = 1.0 - distance
                else:
                    score = -distance  # For L2/IP, lower is better

                doc = LangChainDocument(
                    page_content=text,
                    metadata={
                        **metadata,
                        "similarity_score": score,
                        "distance": distance,
                        "rank": i + 1
                    }
                )
                docs.append(doc)

        return docs

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Get all chunk IDs for this document/user
            where_filter = self._build_where_filter(user_id=user_id, document_id=document_id)
            results = self._collection.get(
                where=where_filter,
                include=["metadatas"]
            )

            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                log.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            return False
        except Exception as e:
            log.error(f"Failed to delete document {document_id}: {e}")
            return False

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Chroma requires a valid where clause; get all IDs and delete by ID
            all_results = self._collection.get(include=[])
            if all_results["ids"]:
                self._collection.delete(ids=all_results["ids"])
            log.warning("Cleared all documents from vector store")
        except Exception as e:
            log.error(f"Failed to clear vector store: {e}")

    def persist(self) -> None:
        """Chroma auto-persists, but this ensures flush."""
        # Chroma PersistentClient auto-persists
        pass

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self._collection.count()
            metadata = self._collection.metadata or {}
            return {
                "collection_name": self.config.collection_name,
                "total_chunks": count,
                "persist_directory": self.config.persist_directory,
                "distance_metric": self.config.distance_metric,
                "embedding_dimension": metadata.get(EMBEDDING_DIMENSION_KEY),
                "embedding_model": metadata.get(EMBEDDING_MODEL_KEY),
            }
        except Exception as e:
            log.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


def create_vector_store(
    config: Optional[VectorStoreConfig] = None,
    embedding_function=None,
    store_type: str = "chroma"
) -> VectorStore:
    """
    Factory function to create vector store instances.
    Supports future Cloudflare Vectorize migration.
    """
    if store_type == "chroma":
        return LocalVectorStore(config, embedding_function)
    elif store_type == "vectorize":
        # Future: Cloudflare Vectorize implementation
        raise NotImplementedError("Cloudflare Vectorize not yet implemented")
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def create_vector_store_from_settings() -> VectorStore:
    """
    Create vector store with configuration from application settings.
    Uses RAG_EMBEDDING_TIER to determine the correct embedding dimension.
    """
    from ..config import get_settings, get_embedding_model_name, get_embedding_dimension
    
    settings = get_settings()
    
    model_name = get_embedding_model_name()
    expected_dim = get_embedding_dimension(model_name)
    
    config = VectorStoreConfig(
        persist_directory=settings.RAG_VECTOR_STORE_DIR,
        collection_name=settings.RAG_COLLECTION_NAME,
        distance_metric=settings.RAG_DISTANCE_METRIC,
        expected_embedding_dimension=expected_dim,
        embedding_model_name=model_name,
    )
    
    return LocalVectorStore(config)


# Export
__all__ = [
    "VectorStore",
    "VectorStoreConfig",
    "LocalVectorStore",
    "create_vector_store",
]