"""
test_rag_vectorstore.py — Vector Store Tests
=============================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.vectorstore import LocalVectorStore, VectorStoreConfig
from backend.rag.embeddings import BGEEmbeddings, EmbeddingConfig
from backend.rag.document_loader import load_document
from backend.rag.chunking import chunk_documents, ChunkConfig


@pytest.fixture
def temp_vector_store(tmp_path):
    config = VectorStoreConfig(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_vs"
    )
    store = LocalVectorStore(config)
    yield store
    store.clear()


@pytest.fixture
def sample_chunks_with_embeddings(sample_pdf):
    loaded = load_document(sample_pdf, "doc-1", "test.pdf", "pdf")
    chunks = chunk_documents(loaded.documents, ChunkConfig(chunk_size=100)).chunks
    embeddings = BGEEmbeddings(EmbeddingConfig())
    chunk_embeddings = embeddings.embed_chunks(chunks)
    return chunks, chunk_embeddings


class TestVectorStoreInitialization:
    def test_initializes_collection(self, temp_vector_store):
        stats = temp_vector_store.get_collection_stats()
        assert stats["collection_name"] == "test_vs"
        assert stats["total_chunks"] == 0

    def test_persist_directory_created(self, tmp_path):
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "new_chroma"),
            collection_name="test"
        )
        store = LocalVectorStore(config)
        assert (tmp_path / "new_chroma").exists()
        store.clear()


class TestAddDocuments:
    def test_add_documents_returns_ids(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        ids = temp_vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            user_id="user-1",
            document_id="doc-1"
        )
        assert len(ids) == len(chunks)
        assert all(isinstance(id, str) for id in ids)

    def test_add_documents_with_metadata(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")
        stats = temp_vector_store.get_collection_stats()
        assert stats["total_chunks"] == len(chunks)

    def test_add_empty_list(self, temp_vector_store):
        ids = temp_vector_store.add_documents([], [], "user-1", "doc-1")
        assert ids == []


class TestSimilaritySearch:
    def test_search_returns_results(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")

        query_emb = BGEEmbeddings(EmbeddingConfig()).embed_query("machine learning")
        results = temp_vector_store.similarity_search(query_emb, k=3, user_id="user-1")

        assert len(results) >= 1
        assert len(results) <= 3
        for doc in results:
            assert "similarity_score" in doc.metadata
            assert "distance" in doc.metadata
            assert "rank" in doc.metadata

    def test_search_respects_k(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")

        query_emb = BGEEmbeddings(EmbeddingConfig()).embed_query("machine learning")
        results = temp_vector_store.similarity_search(query_emb, k=1, user_id="user-1")
        assert len(results) <= 1

    def test_search_filters_by_user(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")
        temp_vector_store.add_documents(chunks, embeddings, "user-2", "doc-2")

        query_emb = BGEEmbeddings(EmbeddingConfig()).embed_query("machine learning")
        results_1 = temp_vector_store.similarity_search(query_emb, k=10, user_id="user-1")
        results_2 = temp_vector_store.similarity_search(query_emb, k=10, user_id="user-2")

        assert all(r.metadata["user_id"] == "user-1" for r in results_1)
        assert all(r.metadata["user_id"] == "user-2" for r in results_2)

    def test_search_filters_by_document(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-2")

        query_emb = BGEEmbeddings(EmbeddingConfig()).embed_query("machine learning")
        results = temp_vector_store.similarity_search(query_emb, k=10, user_id="user-1", document_id="doc-1")

        assert all(r.metadata["document_id"] == "doc-1" for r in results)


class TestDeleteDocument:
    def test_delete_document(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")

        deleted = temp_vector_store.delete_document("doc-1", "user-1")
        assert deleted is True

        stats = temp_vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0

    def test_delete_nonexistent(self, temp_vector_store):
        deleted = temp_vector_store.delete_document("nonexistent", "user-1")
        assert deleted is False


class TestClear:
    def test_clear_removes_all(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-2")

        temp_vector_store.clear()
        stats = temp_vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0


class TestPersist:
    def test_persist_does_not_error(self, temp_vector_store, sample_chunks_with_embeddings):
        chunks, embeddings = sample_chunks_with_embeddings
        temp_vector_store.add_documents(chunks, embeddings, "user-1", "doc-1")
        temp_vector_store.persist()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])