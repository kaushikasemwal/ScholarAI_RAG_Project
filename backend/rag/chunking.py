"""
chunking.py — Document Chunking for RAG
========================================
Splits documents into overlapping chunks while preserving metadata.
Uses LangChain's RecursiveCharacterTextSplitter for reliable, reproducible chunking.
"""

from dataclasses import dataclass
from typing import List, Optional
import hashlib

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    # Fallback implementations if langchain not available
    LangChainDocument = None
    RecursiveCharacterTextSplitter = None

from .document_loader import LangChainDocument as LoaderDocument, LoadedDocument


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    add_start_index: bool = True

    def __post_init__(self):
        if self.separators is None:
            # Default separators optimized for educational content
            self.separators = [
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "? ",        # Question endings
                "! ",        # Exclamation endings
                "; ",        # Semicolons
                ": ",        # Colons
                ", ",        # Commas
                " ",         # Spaces
                "",          # Character-level fallback
            ]


@dataclass
class ChunkResult:
    """Result of chunking operation."""
    chunks: List[LangChainDocument]
    total_chunks: int
    config: ChunkConfig
    source_document_id: str


def generate_chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    """
    Generate a deterministic chunk ID based on document, page, and content.
    Uses a short hash of the text for uniqueness.
    """
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{document_id}_p{page_number}_c{chunk_index}_{content_hash}"


def chunk_documents(
    documents: List[LangChainDocument],
    config: Optional[ChunkConfig] = None
) -> ChunkResult:
    """
    Split documents into overlapping chunks while preserving metadata.

    Each chunk inherits metadata from its source document and adds:
    - chunk_id: deterministic unique identifier
    - chunk_index: index within the page
    - start_index: character offset in original text
    - chunk_type: "chunk" (vs "page" for unchunked)
    """
    if config is None:
        config = ChunkConfig()

    if RecursiveCharacterTextSplitter is None:
        raise ImportError(
            "langchain_text_splitters not installed. "
            "Install with: pip install langchain-text-splitters"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        keep_separator=config.keep_separator,
        add_start_index=config.add_start_index,
    )

    all_chunks = []

    for doc in documents:
        # Only chunk non-empty documents
        if not doc.page_content.strip():
            continue

        page_number = doc.metadata.get("page_number", 1)
        document_id = doc.metadata.get("document_id", "unknown")

        # Split the document
        chunks = splitter.split_documents([doc])

        # Add chunk-specific metadata
        for chunk_idx, chunk in enumerate(chunks):
            # Generate deterministic chunk ID
            chunk_id = generate_chunk_id(
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_idx,
                text=chunk.page_content
            )

            # Preserve original metadata and add chunk metadata
            chunk.metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "chunk_type": "chunk",
                "parent_page_number": page_number,
            }
            # Ensure start_index is preserved from splitter
            if "start_index" not in chunk.metadata and config.add_start_index:
                chunk.metadata["start_index"] = chunk_idx * (config.chunk_size - config.chunk_overlap)

            all_chunks.append(chunk)

    return ChunkResult(
        chunks=all_chunks,
        total_chunks=len(all_chunks),
        config=config,
        source_document_id=documents[0].metadata.get("document_id", "unknown") if documents else "unknown"
    )


def chunk_loaded_document(
    loaded_doc: LoadedDocument,
    config: Optional[ChunkConfig] = None
) -> ChunkResult:
    """Convenience function to chunk a LoadedDocument."""
    return chunk_documents(loaded_doc.documents, config)


def chunk_by_page(documents: List[LangChainDocument]) -> List[LangChainDocument]:
    """
    Alternative: return documents as-is (one chunk per page/slide).
    Useful for short documents where page-level granularity is sufficient.
    """
    result = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        doc.metadata = {
            **doc.metadata,
            "chunk_type": "page",
            "chunk_index": 0,
        }
        result.append(doc)
    return result


# Export
__all__ = [
    "ChunkConfig",
    "ChunkResult",
    "chunk_documents",
    "chunk_loaded_document",
    "chunk_by_page",
    "generate_chunk_id",
]