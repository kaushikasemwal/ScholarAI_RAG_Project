"""
test_rag_chunking.py — Chunking Tests
======================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.chunking import (
    chunk_documents,
    ChunkConfig,
    generate_chunk_id,
    chunk_by_page,
)
from backend.rag.document_loader import load_document, LoadedDocument


class TestChunkConfig:
    def test_default_config(self):
        config = ChunkConfig()
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators is not None
        assert len(config.separators) > 0

    def test_custom_config(self):
        config = ChunkConfig(chunk_size=200, chunk_overlap=20)
        assert config.chunk_size == 200
        assert config.chunk_overlap == 20


class TestGenerateChunkId:
    def test_deterministic(self):
        id1 = generate_chunk_id("doc-1", 1, 0, "test content")
        id2 = generate_chunk_id("doc-1", 1, 0, "test content")
        assert id1 == id2

    def test_different_content_different_id(self):
        id1 = generate_chunk_id("doc-1", 1, 0, "content a")
        id2 = generate_chunk_id("doc-1", 1, 0, "content b")
        assert id1 != id2

    def test_different_page_different_id(self):
        id1 = generate_chunk_id("doc-1", 1, 0, "same content")
        id2 = generate_chunk_id("doc-1", 2, 0, "same content")
        assert id1 != id2


class TestChunkDocuments:
    @pytest.fixture
    def loaded_doc(self, sample_pdf):
        return load_document(
            sample_pdf,
            file_id="test-doc",
            filename="test.pdf",
            file_type="pdf"
        )

    def test_creates_chunks(self, loaded_doc):
        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        result = chunk_documents(loaded_doc.documents, config)
        assert result.total_chunks >= 1
        assert len(result.chunks) == result.total_chunks

    def test_chunk_metadata(self, loaded_doc):
        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        result = chunk_documents(loaded_doc.documents, config)
        for chunk in result.chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_type" in chunk.metadata
            assert chunk.metadata["chunk_type"] == "chunk"
            assert "document_id" in chunk.metadata
            assert chunk.metadata["document_id"] == "test-doc"

    def test_chunk_overlap_creates_more_chunks(self, loaded_doc):
        config_small = ChunkConfig(chunk_size=100, chunk_overlap=10)
        config_large = ChunkConfig(chunk_size=100, chunk_overlap=50)
        result_small = chunk_documents(loaded_doc.documents, config_small)
        result_large = chunk_documents(loaded_doc.documents, config_large)
        # More overlap = more chunks for same content
        assert result_large.total_chunks >= result_small.total_chunks

    def test_preserves_page_metadata(self, loaded_doc):
        config = ChunkConfig(chunk_size=100, chunk_overlap=20)
        result = chunk_documents(loaded_doc.documents, config)
        for chunk in result.chunks:
            assert "parent_page_number" in chunk.metadata
            assert "page_number" in chunk.metadata


class TestChunkByPage:
    def test_returns_same_count_as_input(self, sample_pdf):
        loaded = load_document(
            sample_pdf,
            file_id="test-doc",
            filename="test.pdf",
            file_type="pdf"
        )
        result = chunk_by_page(loaded.documents)
        assert len(result) == len(loaded.documents)
        for doc in result:
            assert doc.metadata["chunk_type"] == "page"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])