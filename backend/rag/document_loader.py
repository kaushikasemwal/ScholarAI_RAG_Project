"""
document_loader.py — Structured Document Loading for RAG
========================================================
Converts StructuredDocument (from utils) into LangChain Document objects
with rich metadata for vector storage and retrieval.
"""

from typing import List, Optional
from dataclasses import dataclass

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    # Fallback if langchain not available
    from dataclasses import dataclass
    @dataclass
    class LangChainDocument:
        page_content: str
        metadata: dict

from ..utils import (
    StructuredDocument,
    DocumentPage,
    extract_structured_document,
    extract_pdf_pages,
    extract_pptx_slides,
)


@dataclass
class LoadedDocument:
    """Container for loaded document with LangChain documents and metadata."""
    file_id: str
    filename: str
    file_type: str
    documents: List[LangChainDocument]
    total_pages: int
    metadata: dict


def page_to_langchain_doc(
    page: DocumentPage,
    file_id: str,
    filename: str,
    file_type: str,
    chunk_id: Optional[int] = None
) -> LangChainDocument:
    """
    Convert a DocumentPage to a LangChain Document with metadata.

    Metadata includes:
    - document_id: unique file identifier
    - source_filename: original filename
    - file_type: pdf or pptx
    - page_number: 1-indexed page/slide number
    - chunk_id: optional chunk identifier within the page
    - chunk_type: "page" or "chunk" for retrieval filtering
    """
    metadata = {
        "document_id": file_id,
        "source_filename": filename,
        "file_type": file_type,
        "page_number": page.page_number,
        "chunk_type": "page",
        "extraction_source": page.metadata.get("source", "unknown"),
    }
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
        metadata["chunk_type"] = "chunk"

    return LangChainDocument(
        page_content=page.text,
        metadata=metadata
    )


def load_document(
    raw_bytes: bytes,
    file_id: str,
    filename: str,
    file_type: str
) -> LoadedDocument:
    """
    Load a document from raw bytes into LangChain Documents.

    This is the main entry point for RAG document loading.
    Preserves page/slide boundaries as individual documents.
    """
    structured = extract_structured_document(raw_bytes, file_id, filename, file_type)

    documents = [
        page_to_langchain_doc(page, file_id, filename, file_type)
        for page in structured.pages
    ]

    return LoadedDocument(
        file_id=file_id,
        filename=filename,
        file_type=file_type,
        documents=documents,
        total_pages=structured.total_pages,
        metadata=structured.metadata
    )


def load_pdf(raw_bytes: bytes, file_id: str, filename: str) -> LoadedDocument:
    """Load a PDF document."""
    return load_document(raw_bytes, file_id, filename, "pdf")


def load_pptx(raw_bytes: bytes, file_id: str, filename: str) -> LoadedDocument:
    """Load a PPTX/PPT document."""
    return load_document(raw_bytes, file_id, filename, "pptx")


def load_from_file_id(
    file_id: str,
    user_id: str,
    storage_backend=None
) -> LoadedDocument:
    """
    Load document from storage using file_id.
    Requires storage backend to retrieve encrypted bytes.
    """
    if storage_backend is None:
        from ..storage import get_storage
        storage_backend = get_storage()

    from ..utils import decrypt_file
    from ..app import FILE_STORE, FILE_STORE_LOCK

    with FILE_STORE_LOCK:
        if file_id not in FILE_STORE:
            raise ValueError(f"File {file_id} not found in FILE_STORE")
        meta = FILE_STORE[file_id]

    if meta.get("user_id") != user_id:
        raise PermissionError(f"User {user_id} does not own file {file_id}")

    storage_key = meta["storage_key"]
    ext = meta["ext"]
    filename = meta["filename"]

    encrypted_data = storage_backend.get_object(storage_key)
    raw_bytes = decrypt_file(encrypted_data)

    file_type = ext.lstrip(".")
    return load_document(raw_bytes, file_id, filename, file_type)


# Export for backward compatibility
__all__ = [
    "load_document",
    "load_pdf",
    "load_pptx",
    "load_from_file_id",
    "page_to_langchain_doc",
    "LoadedDocument",
    "LangChainDocument",
]