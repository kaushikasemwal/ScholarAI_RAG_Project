"""
test_rag_dimension_validation.py — Dimension Validation Tests
===============================================================
Tests for ChromaDB embedding dimension mismatch protection.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch

from backend.rag.vectorstore import LocalVectorStore, VectorStoreConfig
from backend.rag.embeddings import BGEEmbeddings, EmbeddingConfig


class TestDimensionValidation:
    """Test embedding dimension validation in vector store."""

    def test_collection_stores_embedding_metadata(self, tmp_path):
        """Collection should store embedding dimension and model in metadata."""
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_dim_meta",
            expected_embedding_dimension=768,
            embedding_model_name="BAAI/bge-base-en-v1.5"
        )
        store = LocalVectorStore(config)
        
        stats = store.get_collection_stats()
        assert stats["embedding_dimension"] == 768
        assert stats["embedding_model"] == "BAAI/bge-base-en-v1.5"

    def test_collection_stores_small_embedding_metadata(self, tmp_path):
        """Collection should store 384 dimension for BGE-small."""
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_dim_meta_small",
            expected_embedding_dimension=384,
            embedding_model_name="BAAI/bge-small-en-v1.5"
        )
        store = LocalVectorStore(config)
        
        stats = store.get_collection_stats()
        assert stats["embedding_dimension"] == 384
        assert stats["embedding_model"] == "BAAI/bge-small-en-v1.5"

    def test_insert_validates_embedding_dimension(self, tmp_path):
        """Insert should validate embedding dimension matches config."""
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_validate_dim",
            expected_embedding_dimension=768,
            embedding_model_name="BAAI/bge-base-en-v1.5"
        )
        store = LocalVectorStore(config)
        
        # Create mock documents
        from langchain_core.documents import Document
        docs = [Document(page_content="test content", metadata={"chunk_id": "test_1"})]
        
        # 768-dim embeddings should work
        embeddings_768 = [[0.1] * 768]
        ids = store.add_documents(docs, embeddings_768, "user-1", "doc-1")
        assert len(ids) == 1
        
        # 384-dim embeddings should fail
        embeddings_384 = [[0.1] * 384]
        with pytest.raises(ValueError, match="Embedding dimension mismatch on insert"):
            store.add_documents(docs, embeddings_384, "user-1", "doc-1")

    def test_mismatch_error_message_contains_details(self, tmp_path):
        """Error message should contain configured and actual dimensions."""
        config = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_error_msg",
            expected_embedding_dimension=768,
            embedding_model_name="BAAI/bge-base-en-v1.5"
        )
        store = LocalVectorStore(config)
        
        from langchain_core.documents import Document
        docs = [Document(page_content="test", metadata={"chunk_id": "t"})]
        embeddings_384 = [[0.1] * 384]
        
        with pytest.raises(ValueError) as exc_info:
            store.add_documents(docs, embeddings_384, "user-1", "doc-1")
        
        error_msg = str(exc_info.value)
        assert "Expected dimension: 768" in error_msg
        assert "Actual embedding dimension: 384" in error_msg
        assert "BAAI/bge-base-en-v1.5" in error_msg
        assert "Resolution options" in error_msg


class TestCollectionDimensionMismatch:
    """Test handling of existing collections with different dimensions."""

    def test_existing_collection_wrong_dimension_raises(self, tmp_path):
        """Creating store with different dimension than existing collection should raise."""
        # First, create a collection with 768 dimensions
        config_768 = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="shared_collection",
            expected_embedding_dimension=768,
            embedding_model_name="BAAI/bge-base-en-v1.5"
        )
        store_768 = LocalVectorStore(config_768)
        
        # Add a document to ensure collection exists
        from langchain_core.documents import Document
        docs = [Document(page_content="test", metadata={"chunk_id": "t"})]
        store_768.add_documents(docs, [[0.1] * 768], "user-1", "doc-1")
        
        # Now try to create another store with 384 dimensions for same collection
        config_384 = VectorStoreConfig(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="shared_collection",
            expected_embedding_dimension=384,
            embedding_model_name="BAAI/bge-small-en-v1.5"
        )
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch detected"):
            LocalVectorStore(config_384)

    def test_legacy_collection_without_metadata_warns(self, tmp_path, caplog):
        """Legacy collection without dimension metadata should warn but not fail."""
        import chromadb
        from chromadb.config import Settings
        
        # Manually create a collection without dimension metadata
        client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma_test"),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        collection = client.create_collection(
            name="legacy_collection",
            metadata={"hnsw:space": "cosine"}  # No embedding_dimension or embedding_model
        )
        collection.add(
            ids=["test_1"],
            embeddings=[[0.1] * 768],
            documents=["test content"],
            metadatas=[{"user_id": "user-1", "document_id": "doc-1", "chunk_id": "test_1"}]
        )
        
        # Now create store - should warn but not fail
        with caplog.at_level("WARNING"):
            config = VectorStoreConfig(
                persist_directory=str(tmp_path / "chroma_test"),
                collection_name="legacy_collection",
                expected_embedding_dimension=768,
                embedding_model_name="BAAI/bge-base-en-v1.5"
            )
            store = LocalVectorStore(config)
            
            # Should have logged a warning about missing metadata
            assert any("no embedding dimension metadata" in msg for msg in caplog.messages)


class TestBGEEmbeddingsDimension:
    """Test BGEEmbeddings produces correct dimensions."""

    def test_bge_base_produces_768_dim(self):
        """BGE-base config should produce 768-dim embeddings."""
        import numpy as np
        config = EmbeddingConfig(
            model_name="BAAI/bge-base-en-v1.5",
            expected_dimension=768
        )
        embeddings = BGEEmbeddings(config)
        
        # Mock the internal model attribute directly
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 768])
        embeddings._model = mock_model
        
        vec = embeddings.embed_query("test")
        assert len(vec) == 768

    def test_bge_small_produces_384_dim(self):
        """BGE-small config should produce 384-dim embeddings."""
        import numpy as np
        config = EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            expected_dimension=384
        )
        embeddings = BGEEmbeddings(config)
        
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        embeddings._model = mock_model
        
        vec = embeddings.embed_query("test")
        assert len(vec) == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])