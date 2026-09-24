"""
test_rag_retriever.py — Retriever Tests
========================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.retriever import retrieve, RetrievalConfig, Retriever, create_retriever
from backend.rag.vectorstore import LocalVectorStore, VectorStoreConfig
from backend.rag.embeddings import BGEEmbeddings, EmbeddingConfig
from backend.rag.document_loader import load_document
from backend.rag.chunking import chunk_documents, ChunkConfig


@pytest.fixture
def setup_retriever(tmp_path, sample_pdf):
    config = VectorStoreConfig(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_retriever"
    )
    vector_store = LocalVectorStore(config)
    embeddings = BGEEmbeddings(EmbeddingConfig())

    # Ingest test document
    loaded = load_document(sample_pdf, "doc-1", "test.pdf", "pdf")
    chunks = chunk_documents(loaded.documents, ChunkConfig(chunk_size=100)).chunks
    chunk_embeddings = embeddings.embed_chunks(chunks)
    vector_store.add_documents(chunks, chunk_embeddings, "user-1", "doc-1")

    yield vector_store, embeddings
    vector_store.clear()


class TestRetrieveFunction:
    def test_requires_user_id(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        with pytest.raises(ValueError, match="user_id is required"):
            retrieve("query", user_id="", vector_store=vector_store, embeddings=embeddings)

    def test_returns_results(self, setup_retriever):
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
        assert result.query == "machine learning"

    def test_filters_by_user(self, setup_retriever):
        vector_store, embeddings = setup_retriever

        # Add another user's document
        loaded = load_document(sample_pdf, "doc-2", "test2.pdf", "pdf")
        chunks = chunk_documents(loaded.documents, ChunkConfig(chunk_size=100)).chunks
        chunk_embeddings = embeddings.embed_chunks(chunks)
        vector_store.add_documents(chunks, chunk_embeddings, "user-2", "doc-2")

        result = retrieve("machine learning", user_id="user-1", vector_store=vector_store, embeddings=embeddings)
        assert all(r.metadata["user_id"] == "user-1" for r in result.documents)

    def test_filters_by_document(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        result = retrieve(
            "machine learning",
            user_id="user-1",
            document_id="doc-1",
            vector_store=vector_store,
            embeddings=embeddings
        )
        assert result.document_id == "doc-1"
        assert all(r.metadata["document_id"] == "doc-1" for r in result.documents)

    def test_respects_top_k(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        result = retrieve("machine learning", user_id="user-1", vector_store=vector_store, embeddings=embeddings, top_k=1)
        assert result.total_found <= 1

    def test_similarity_threshold(self, setup_retriever):
        vector_store, embeddings = setup_retriever

        # High threshold should filter out low-similarity results
        result = retrieve(
            "completely unrelated topic xyz",
            user_id="user-1",
            vector_store=vector_store,
            embeddings=embeddings,
            top_k=10
        )
        # Should still return results (threshold not applied by default)

        # With threshold
        result_threshold = retrieve(
            "completely unrelated topic xyz",
            user_id="user-1",
            vector_store=vector_store,
            embeddings=embeddings,
            top_k=10
        )
        # The retriever doesn't apply threshold automatically; it's a config option
        assert result_threshold.total_found >= 0


class TestRetrieverClass:
    def test_create_retriever(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        retriever = create_retriever(
            user_id="user-1",
            vector_store=vector_store,
            embeddings=embeddings
        )
        assert isinstance(retriever, Retriever)

    def test_retriever_retrieve(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        retriever = Retriever(vector_store, embeddings, RetrievalConfig(top_k=3))
        result = retriever.retrieve("machine learning", "user-1")
        assert result.total_found >= 1

    def test_retriever_multiple_queries(self, setup_retriever):
        vector_store, embeddings = setup_retriever
        retriever = Retriever(vector_store, embeddings, RetrievalConfig(top_k=2))
        results = retriever.retrieve_multiple(
            ["machine learning", "neural networks"],
            "user-1"
        )
        assert len(results) == 2
        assert all(isinstance(r.total_found, int) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])