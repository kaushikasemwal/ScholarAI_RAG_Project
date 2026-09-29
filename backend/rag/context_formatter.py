"""
context_formatter.py — Context Formatting for RAG Prompts
==========================================================
Deterministic formatting of retrieved chunks into prompt context.
"""

from typing import List, Optional
from dataclasses import dataclass

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None


@dataclass
class ContextFormatConfig:
    """Configuration for context formatting."""
    include_page: bool = True
    include_chunk_id: bool = False  # Disable by default to reduce prompt noise
    include_score: bool = False
    include_source_filename: bool = False
    max_chars_per_chunk: int = 1000
    separator: str = "\n\n---\n\n"
    chunk_template: str = "[Source {index}]\nPage/Slide: {page}\nContent:\n{content}"


def create_generation_context_config() -> ContextFormatConfig:
    """
    Create a context config optimized for grounded question generation.
    
    Minimal metadata to reduce noise while preserving page/slide attribution.
    """
    return ContextFormatConfig(
        include_page=True,
        include_chunk_id=False,
        include_score=False,
        include_source_filename=False,
        max_chars_per_chunk=800,  # Slightly smaller for generation context
        separator="\n\n---\n\n",
    )


def format_chunk(
    doc: "LangChainDocument",
    index: int,
    config: Optional[ContextFormatConfig] = None
) -> str:
    """
    Format a single retrieved chunk for inclusion in the prompt.
    
    Args:
        doc: LangChain Document with metadata
        index: 1-based index of this chunk in the retrieved set
        config: Formatting configuration
    
    Returns:
        Formatted chunk string
    """
    if config is None:
        config = ContextFormatConfig()
    
    metadata = doc.metadata
    content = doc.page_content
    
    # Truncate content if too long
    if len(content) > config.max_chars_per_chunk:
        content = content[:config.max_chars_per_chunk] + "... [truncated]"
    
    page = metadata.get("page_number", metadata.get("slide_number", "Unknown"))
    chunk_id = metadata.get("chunk_id", f"chunk_{index}")
    
    # Build the formatted chunk based on config
    parts = [f"[Source {index}]"]
    
    if config.include_page:
        parts.append(f"Page/Slide: {page}")
    
    if config.include_chunk_id:
        parts.append(f"Chunk ID: {chunk_id}")
    
    if config.include_score and "similarity_score" in metadata:
        parts.append(f"Relevance: {metadata['similarity_score']:.3f}")
    
    if config.include_source_filename and "source_filename" in metadata:
        parts.append(f"Source: {metadata['source_filename']}")
    
    parts.append(f"Content:\n{content}")
    
    return "\n".join(parts)


def format_retrieved_context(
    documents: List["LangChainDocument"],
    config: Optional[ContextFormatConfig] = None
) -> str:
    """
    Format a list of retrieved documents into a single context string for the prompt.
    
    Args:
        documents: List of retrieved LangChain Documents
        config: Formatting configuration
    
    Returns:
        Formatted context string
    """
    if config is None:
        config = ContextFormatConfig()
    
    if not documents:
        return "[No relevant context retrieved]"
    
    formatted_chunks = []
    for i, doc in enumerate(documents, 1):
        formatted_chunks.append(format_chunk(doc, i, config))
    
    return config.separator.join(formatted_chunks)


def extract_provenance(
    documents: List["LangChainDocument"]
) -> List[dict]:
    """
    Extract provenance information from retrieved documents.
    
    Args:
        documents: List of retrieved LangChain Documents
    
    Returns:
        List of provenance dictionaries
    """
    provenance = []
    for doc in documents:
        metadata = doc.metadata
        page_num = metadata.get("page_number", metadata.get("slide_number"))
        provenance.append({
            "document_id": metadata.get("document_id"),
            "source_filename": metadata.get("source_filename"),
            "file_type": metadata.get("file_type"),
            "page_number": page_num,
            "page_or_slide": page_num,  # Backward compatibility
            "chunk_id": metadata.get("chunk_id"),
            "chunk_type": metadata.get("chunk_type"),
            "similarity_score": metadata.get("similarity_score"),
            "rank": metadata.get("rank"),
        })
    return provenance


def format_context_for_prompt(
    documents: List["LangChainDocument"],
    max_chunks: Optional[int] = None,
    config: Optional[ContextFormatConfig] = None
) -> str:
    """
    High-level function to format retrieved context for prompt injection.
    Applies chunk limit if specified.
    
    Args:
        documents: Retrieved documents
        max_chunks: Maximum number of chunks to include
        config: Formatting configuration
    
    Returns:
        Formatted context string ready for prompt
    """
    if max_chunks is not None:
        documents = documents[:max_chunks]
    
    return format_retrieved_context(documents, config)


# Export
__all__ = [
    "ContextFormatConfig",
    "format_chunk",
    "format_retrieved_context",
    "extract_provenance",
    "format_context_for_prompt",
    "create_generation_context_config",
]