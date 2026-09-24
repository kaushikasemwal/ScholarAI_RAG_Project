"""
pipeline.py — RAG Ingestion Pipeline
=====================================
Orchestrates the full document ingestion flow:
load → chunk → embed → store
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

from .document_loader import load_document, LoadedDocument
from .chunking import chunk_documents, ChunkConfig, ChunkResult
from .embeddings import BGEEmbeddings, EmbeddingConfig, get_embeddings
from .vectorstore import VectorStore, LocalVectorStore, VectorStoreConfig, create_vector_store

log = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    chunk_config: Optional[ChunkConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None
    vector_store_type: str = "chroma"
    skip_empty_chunks: bool = True


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    file_id: str
    user_id: str
    filename: str
    file_type: str
    total_pages: int
    total_chunks: int
    chunk_ids: List[str]
    vector_store_stats: Dict[str, Any]
    config: IngestionConfig


def ingest_document(
    raw_bytes: bytes,
    file_id: str,
    user_id: str,
    filename: str,
    file_type: str,
    config: Optional[IngestionConfig] = None,
    vector_store: Optional[VectorStore] = None,
    embeddings: Optional[BGEEmbeddings] = None,
) -> IngestionResult:
    """
    Ingest a document into the RAG pipeline.

    Pipeline:
    1. Load document with structured page/slide metadata
    2. Chunk documents with overlap preserving metadata
    3. Generate embeddings for each chunk
    4. Store in vector store with user/document isolation

    Args:
        raw_bytes: Raw file bytes
        file_id: Unique document identifier
        user_id: User ID for isolation
        filename: Original filename
        file_type: "pdf" or "pptx"
        config: Ingestion configuration
        vector_store: Optional pre-configured vector store
        embeddings: Optional pre-configured embeddings

    Returns:
        IngestionResult with statistics
    """
    if config is None:
        config = IngestionConfig()

    # Initialize dependencies if not provided
    if vector_store is None:
        vector_store = create_vector_store(config.vector_store_config, store_type=config.vector_store_type)

    if embeddings is None:
        embeddings = get_embeddings(config.embedding_config)

    # Step 1: Load document with structured metadata
    log.info(f"Loading document {file_id} ({filename})")
    loaded_doc = load_document(raw_bytes, file_id, filename, file_type)

    if not loaded_doc.documents:
        log.warning(f"Document {file_id} has no extractable content")
        return IngestionResult(
            file_id=file_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            total_pages=0,
            total_chunks=0,
            chunk_ids=[],
            vector_store_stats={},
            config=config
        )

    # Step 2: Chunk documents
    log.info(f"Chunking document {file_id} ({loaded_doc.total_pages} pages)")
    chunk_result = chunk_documents(loaded_doc.documents, config.chunk_config)

    chunks = chunk_result.chunks

    if config.skip_empty_chunks:
        chunks = [c for c in chunks if c.page_content.strip()]
        log.info(f"Filtered to {len(chunks)} non-empty chunks")

    if not chunks:
        log.warning(f"Document {file_id} produced no valid chunks")
        return IngestionResult(
            file_id=file_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            total_pages=loaded_doc.total_pages,
            total_chunks=0,
            chunk_ids=[],
            vector_store_stats={},
            config=config
        )

    # Step 3: Generate embeddings
    log.info(f"Generating embeddings for {len(chunks)} chunks")
    chunk_embeddings = embeddings.embed_chunks(chunks)

    # Verify embedding dimension
    if chunk_embeddings and len(chunk_embeddings[0]) != embeddings.get_dimension():
        log.warning(
            f"Embedding dimension mismatch: expected {embeddings.get_dimension()}, "
            f"got {len(chunk_embeddings[0])}"
        )

    # Step 4: Store in vector store
    log.info(f"Storing {len(chunks)} chunks in vector store")
    chunk_ids = vector_store.add_documents(
        documents=chunks,
        embeddings=chunk_embeddings,
        user_id=user_id,
        document_id=file_id
    )

    # Get stats
    stats = vector_store.get_collection_stats()

    result = IngestionResult(
        file_id=file_id,
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        total_pages=loaded_doc.total_pages,
        total_chunks=len(chunks),
        chunk_ids=chunk_ids,
        vector_store_stats=stats,
        config=config
    )

    log.info(f"Successfully ingested document {file_id}: {len(chunks)} chunks, {loaded_doc.total_pages} pages")
    return result


def ingest_from_file_id(
    file_id: str,
    user_id: str,
    config: Optional[IngestionConfig] = None,
    vector_store: Optional[VectorStore] = None,
    embeddings: Optional[BGEEmbeddings] = None,
) -> IngestionResult:
    """
    Ingest a document using file_id from storage.
    Retrieves encrypted bytes from storage backend.
    """
    from ..utils import decrypt_file
    from ..app import FILE_STORE, FILE_STORE_LOCK
    from ..storage import get_storage

    with FILE_STORE_LOCK:
        if file_id not in FILE_STORE:
            raise ValueError(f"File {file_id} not found in FILE_STORE")
        meta = FILE_STORE[file_id]

    if meta.get("user_id") != user_id:
        raise PermissionError(f"User {user_id} does not own file {file_id}")

    storage = get_storage()
    encrypted_data = storage.get_object(meta["storage_key"])
    raw_bytes = decrypt_file(encrypted_data)

    return ingest_document(
        raw_bytes=raw_bytes,
        file_id=file_id,
        user_id=user_id,
        filename=meta["filename"],
        file_type=meta["ext"].lstrip("."),
        config=config,
        vector_store=vector_store,
        embeddings=embeddings,
    )


def delete_document(
    file_id: str,
    user_id: str,
    vector_store: Optional[VectorStore] = None,
    vector_store_config: Optional[VectorStoreConfig] = None,
) -> bool:
    """Delete a document from the vector store."""
    if vector_store is None:
        vector_store = create_vector_store(vector_store_config)
    return vector_store.delete_document(file_id, user_id)


# Export
__all__ = [
    "IngestionConfig",
    "IngestionResult",
    "ingest_document",
    "ingest_from_file_id",
    "delete_document",
]