"""
backend/rag — RAG Pipeline Foundation
======================================
Phase 1: Document ingestion, chunking, embeddings, vector store, retrieval.
Phase 2: Grounded generation with FLAN-T5, prompt templates, output parsing.
Phase 2 Integration: Quiz adapter for ScholarAI API compatibility.
Phase 2 Multi-Question: Multi-question RAG quiz generation with diversity.
"""

from .document_loader import load_document, extract_pdf_pages, extract_pptx_slides
from .chunking import chunk_documents, ChunkConfig
from .embeddings import get_embeddings, EmbeddingConfig, BGEEmbeddings
from .vectorstore import VectorStore, LocalVectorStore, VectorStoreConfig
from .retriever import Retriever, RetrievalConfig, retrieve, create_retriever
from .pipeline import ingest_document, IngestionConfig, ingest_from_file_id
from .prompts import format_prompt, GROUNDED_QG_SYSTEM_PROMPT, PROMPT_VERSION
from .context_formatter import format_context_for_prompt, extract_provenance, ContextFormatConfig
from .output_parser import (
    GroundedQuizQuestion,
    ParseResult,
    GroundedOutputParser,
    create_grounded_output_parser,
)
from .generator import (
    GeneratorConfig,
    GroundedQuestionResult,
    GroundedQuizGenerator,
    create_grounded_generator,
    generate_grounded_question,
)
from .quiz_adapter import (
    RAGQuizQuestion,
    convert_grounded_to_quiz_question,
    convert_multiple_grounded_to_quiz,
    extract_quiz_questions,
    extract_provenance_map,
    rag_questions_to_api_format,
)
from .multi_question import (
    MultiQuestionConfig,
    MultiQuestionRAGGenerator,
    DiversityFilter,
    ChunkTracker,
    create_multi_question_generator,
    generate_rag_quiz,
)

__all__ = [
    # Phase 1
    "load_document",
    "extract_pdf_pages",
    "extract_pptx_slides",
    "chunk_documents",
    "ChunkConfig",
    "get_embeddings",
    "EmbeddingConfig",
    "BGEEmbeddings",
    "VectorStore",
    "LocalVectorStore",
    "VectorStoreConfig",
    "Retriever",
    "RetrievalConfig",
    "retrieve",
    "create_retriever",
    "ingest_document",
    "ingest_from_file_id",
    "IngestionConfig",
    # Phase 2
    "format_prompt",
    "GROUNDED_QG_SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "format_context_for_prompt",
    "extract_provenance",
    "ContextFormatConfig",
    "GroundedQuizQuestion",
    "ParseResult",
    "GroundedOutputParser",
    "create_grounded_output_parser",
    "GeneratorConfig",
    "GroundedQuestionResult",
    "GroundedQuizGenerator",
    "create_grounded_generator",
    "generate_grounded_question",
    # Phase 2 Integration
    "RAGQuizQuestion",
    "convert_grounded_to_quiz_question",
    "convert_multiple_grounded_to_quiz",
    "extract_quiz_questions",
    "extract_provenance_map",
    "rag_questions_to_api_format",
    # Phase 2 Multi-Question
    "MultiQuestionConfig",
    "MultiQuestionRAGGenerator",
    "DiversityFilter",
    "ChunkTracker",
    "create_multi_question_generator",
    "generate_rag_quiz",
]

__version__ = "2.0.0"