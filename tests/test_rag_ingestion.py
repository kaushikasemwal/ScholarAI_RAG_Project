"""
test_rag_ingestion.py — RAG Ingestion Pipeline Tests
=====================================================
Tests for the Phase 1 RAG ingestion pipeline.
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.pipeline import ingest_document, IngestionConfig
from backend.rag.document_loader import load_document, LoadedDocument
from backend.rag.chunking import chunk_documents, ChunkConfig
from backend.rag.embeddings import BGEEmbeddings, EmbeddingConfig
from backend.rag.vectorstore import LocalVectorStore, VectorStoreConfig
from backend.rag.retriever import retrieve, RetrievalConfig


# Sample PDF content for testing
SAMPLE_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Machine learning is a subset of AI.) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000200 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
293
%%EOF"""


class TestStructuredExtraction:
    """Test structured document extraction."""

    def test_extract_pdf_pages_returns_pages(self):
        from backend.utils import extract_pdf_pages, DocumentPage
        pages = extract_pdf_pages(SAMPLE_PDF_CONTENT)
        assert len(pages) >= 1
        assert all(isinstance(p, DocumentPage) for p in pages)
        assert all(p.page_number >= 1 for p in pages)
        assert all(hasattr(p, 'text') for p in pages)

    def test_extract_structured_document(self):
        from backend.utils import extract_structured_document, StructuredDocument
        doc = extract_structured_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )
        assert isinstance(doc, StructuredDocument)
        assert doc.file_id == "test-123"
        assert doc.filename == "test.pdf"
        assert doc.file_type == "pdf"
        assert doc.total_pages >= 1
        assert len(doc.pages) == doc.total_pages


class TestDocumentLoading:
    """Test document loading to LangChain Documents."""

    def test_load_document(self):
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )
        assert isinstance(loaded, LoadedDocument)
        assert loaded.file_id == "test-123"
        assert loaded.filename == "test.pdf"
        assert loaded.file_type == "pdf"
        assert len(loaded.documents) >= 1

        # Check metadata
        doc = loaded.documents[0]
        assert doc.metadata["document_id"] == "test-123"
        assert doc.metadata["source_filename"] == "test.pdf"
        assert doc.metadata["file_type"] == "pdf"
        assert "page_number" in doc.metadata
        assert doc.metadata["chunk_type"] == "page"


class TestChunking:
    """Test document chunking."""

    def test_chunk_documents_creates_chunks(self):
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )

        config = ChunkConfig(chunk_size=200, chunk_overlap=50)
        result = chunk_documents(loaded.documents, config)

        assert result.total_chunks >= 1
        assert all("chunk_id" in c.metadata for c in result.chunks)
        assert all("chunk_index" in c.metadata for c in result.chunks)
        assert all(c.metadata["chunk_type"] == "chunk" for c in result.chunks)

    def test_chunk_ids_are_deterministic(self):
        """Same input should produce same chunk IDs."""
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )

        config = ChunkConfig(chunk_size=200, chunk_overlap=50)
        result1 = chunk_documents(loaded.documents, config)
        result2 = chunk_documents(loaded.documents, config)

        ids1 = [c.metadata["chunk_id"] for c in result1.chunks]
        ids2 = [c.metadata["chunk_id"] for c in result2.chunks]
        assert ids1 == ids2

    def test_metadata_preserved_in_chunks(self):
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )

        config = ChunkConfig(chunk_size=200, chunk_overlap=50)
        result = chunk_documents(loaded.documents, config)

        for chunk in result.chunks:
            assert chunk.metadata["document_id"] == "test-123"
            assert chunk.metadata["source_filename"] == "test.pdf"
            assert chunk.metadata["file_type"] == "pdf"
            assert "parent_page_number" in chunk.metadata


class TestEmbeddings:
    """Test BGE embeddings."""

    def test_embedding_dimension(self):
        """Verify BGE produces 768-dimensional embeddings."""
        embeddings = BGEEmbeddings(EmbeddingConfig())
        assert embeddings.verify_dimension(), "Embedding dimension should be 768"

    def test_embed_query(self):
        embeddings = BGEEmbeddings(EmbeddingConfig())
        vec = embeddings.embed_query("What is machine learning?")
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

    def test_embed_documents(self):
        embeddings = BGEEmbeddings(EmbeddingConfig())
        texts = ["Machine learning is a subset of AI.", "Deep learning uses neural networks."]
        vecs = embeddings.embed_documents(texts)
        assert len(vecs) == 2
        assert all(len(v) == 768 for v in vecs)


class TestVectorStore:
    """Test local vector store."""

    @pytest.fixture
    def temp_vector_store(self, tmp_path):
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection"
        )
        store = LocalVectorStore(config)
        yield store
        # Cleanup
        store.clear()

    def test_add_and_search(self, temp_vector_store):
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )

        config = ChunkConfig(chunk_size=200, chunk_overlap=50)
        chunks = chunk_documents(loaded.documents, config).chunks

        embeddings = BGEEmbeddings(EmbeddingConfig())
        chunk_embeddings = embeddings.embed_chunks(chunks)

        # Add documents
        ids = temp_vector_store.add_documents(
            documents=chunks,
            embeddings=chunk_embeddings,
            user_id="user-1",
            document_id="test-123"
        )
        assert len(ids) == len(chunks)

        # Search
        query_emb = embeddings.embed_query("machine learning")
        results = temp_vector_store.similarity_search(
            query_embedding=query_emb,
            k=3,
            user_id="user-1"
        )
        assert len(results) >= 1
        assert all("similarity_score" in r.metadata for r in results)

    def test_user_isolation(self, temp_vector_store):
        """Verify user isolation works."""
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="doc-1",
            filename="test.pdf",
            file_type="pdf"
        )
        chunks = chunk_documents(loaded.documents, ChunkConfig(chunk_size=200)).chunks
        embeddings = BGEEmbeddings(EmbeddingConfig())
        chunk_embeddings = embeddings.embed_chunks(chunks)

        # Add for user-1
        temp_vector_store.add_documents(chunks, chunk_embeddings, "user-1", "doc-1")
        # Add for user-2 (different document)
        temp_vector_store.add_documents(chunks, chunk_embeddings, "user-2", "doc-2")

        query_emb = embeddings.embed_query("machine learning")

        # User-1 should only see their docs
        results_1 = temp_vector_store.similarity_search(query_emb, k=10, user_id="user-1")
        assert all(r.metadata["user_id"] == "user-1" for r in results_1)

        # User-2 should only see their docs
        results_2 = temp_vector_store.similarity_search(query_emb, k=10, user_id="user-2")
        assert all(r.metadata["user_id"] == "user-2" for r in results_2)


class TestRetriever:
    """Test retriever with user/document isolation."""

    @pytest.fixture
    def setup_retriever(self, tmp_path):
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection"
        )
        vector_store = LocalVectorStore(config)
        embeddings = BGEEmbeddings(EmbeddingConfig())

        # Ingest a test document
        loaded = load_document(
            SAMPLE_PDF_CONTENT,
            file_id="test-123",
            filename="test.pdf",
            file_type="pdf"
        )
        chunks = chunk_documents(loaded.documents, ChunkConfig(chunk_size=200)).chunks
        chunk_embeddings = embeddings.embed_chunks(chunks)
        vector_store.add_documents(chunks, chunk_embeddings, "user-1", "test-123")

        yield vector_store, embeddings

        vector_store.clear()

    def test_retrieve_requires_user_id(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        with pytest.raises(ValueError, match="user_id is required"):
            retrieve("query", user_id="", vector_store=vector_store, embeddings=embeddings)

    def test_retrieve_with_user_id(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        result = retrieve(
            "machine learning",
            user_id="user-1",
            vector_store=vector_store,
            embeddings=embeddings,
            top_k=3
        )
        assert result.total_found >= 1
        assert result.user_id == "user-1"
        assert all(r.metadata["user_id"] == "user-1" for r in result.documents)

    def test_retrieve_with_document_filter(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        result = retrieve(
            "machine learning",
            user_id="user-1",
            document_id="test-123",
            vector_store=vector_store,
            embeddings=embeddings,
            top_k=3
        )
        assert result.total_found >= 1
        assert result.document_id == "test-123"
        assert all(r.metadata["document_id"] == "test-123" for r in result.documents)


class TestIngestionPipeline:
    """Test the full ingestion pipeline."""

    @pytest.fixture
    def temp_pipeline(self, tmp_path):
        config = IngestionConfig(
            chunk_config=ChunkConfig(chunk_size=200, chunk_overlap=50),
            vector_store_config=VectorStoreConfig(
                persist_directory=str(tmp_path / "chroma_test"),
                collection_name="test_pipeline"
            )
        )
        yield config
        # Cleanup handled by vector store fixture

    def test_ingest_document(self, tmp_path):
        config = IngestionConfig(
            chunk_config=ChunkConfig(chunk_size=200, chunk_overlap=50),
            vector_store_config=VectorStoreConfig(
                persist_directory=str(tmp_path / "chroma_test"),
                collection_name="test_pipeline"
            )
        )

        result = ingest_document(
            raw_bytes=SAMPLE_PDF_CONTENT,
            file_id="test-456",
            user_id="user-1",
            filename="test.pdf",
            file_type="pdf",
            config=config
        )

        assert result.file_id == "test-456"
        assert result.user_id == "user-1"
        assert result.total_chunks >= 1
        assert len(result.chunk_ids) == result.total_chunks
        assert result.total_pages >= 1


class TestEmptyDocumentHandling:
    """Test handling of empty/unextractable documents."""

    def test_empty_pdf_handling(self):
        """Test handling of PDF that extracts no text."""
        empty_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
193
%%EOF"""

        from backend.utils import extract_pdf_pages
        pages = extract_pdf_pages(empty_pdf)
        # Should return at least one page (even if empty text)
        assert len(pages) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])