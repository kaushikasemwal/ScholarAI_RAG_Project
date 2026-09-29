"""
test_rag_embeddings.py — Embedding Configuration Tests
=======================================================
Tests for embedding/generation tier separation and dimension validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, Mock

from backend.config import (
    get_settings,
    get_embedding_dimension,
    get_embedding_model_name,
    get_generation_model_name,
    get_summarization_model_name,
)
from backend.rag.embeddings import EmbeddingConfig, create_embedding_config_from_settings
from backend.rag.vectorstore import VectorStoreConfig


class TestEmbeddingDimensionMapping:
    """Test embedding model -> dimension mapping."""

    def test_bge_base_dimension(self):
        """BGE-base should map to 768 dimensions."""
        assert get_embedding_dimension("BAAI/bge-base-en-v1.5") == 768

    def test_bge_small_dimension(self):
        """BGE-small should map to 384 dimensions."""
        assert get_embedding_dimension("BAAI/bge-small-en-v1.5") == 384

    def test_unknown_model_defaults_to_768(self):
        """Unknown models should default to 768 for safety."""
        assert get_embedding_dimension("unknown-model") == 768

    def test_minilm_dimension(self):
        """all-MiniLM-L6-v2 should map to 384 dimensions."""
        assert get_embedding_dimension("all-MiniLM-L6-v2") == 384


class TestEmbeddingTierSelection:
    """Test embedding tier selection is independent of generation tier."""

    def test_default_embedding_tier_quality(self):
        """Default embedding tier should be quality (BGE-base)."""
        settings = get_settings()
        assert settings.RAG_EMBEDDING_TIER == "quality"
        assert settings.RAG_EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert settings.RAG_EMBEDDING_DIMENSION == 768

    def test_embedding_tier_speed_selects_small(self, monkeypatch):
        """Setting embedding tier to speed should select BGE-small."""
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "speed")
        # Reset settings singleton
        from backend.config import reset_settings
        reset_settings()
        
        settings = get_settings()
        assert settings.RAG_EMBEDDING_TIER == "speed"
        assert settings.RAG_EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"  # Base default
        assert settings.RAG_EMBEDDING_MODEL_LIGHT == "BAAI/bge-small-en-v1.5"

    def test_get_embedding_model_name_quality(self):
        """get_embedding_model_name should return base model for quality tier."""
        # Ensure clean settings state
        from backend.config import reset_settings
        reset_settings()
        import os
        os.environ["RAG_EMBEDDING_TIER"] = "quality"
        
        model = get_embedding_model_name()
        assert model == "BAAI/bge-base-en-v1.5"

    def test_get_embedding_model_name_speed(self, monkeypatch):
        """get_embedding_model_name should return small model for speed tier."""
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        model = get_embedding_model_name()
        assert model == "BAAI/bge-small-en-v1.5"

    def test_embedding_dimension_matches_model(self):
        """Embedding dimension should match the selected model."""
        model = get_embedding_model_name()
        dim = get_embedding_dimension(model)
        
        if "bge-base" in model:
            assert dim == 768
        elif "bge-small" in model:
            assert dim == 384


class TestGenerationTierSelection:
    """Test generation tier selection is independent of embedding tier."""

    def test_default_generation_tier_quality(self):
        """Default generation tier should be quality (FLAN-T5-large)."""
        settings = get_settings()
        assert settings.RAG_GENERATION_TIER == "quality"
        assert settings.RAG_GENERATION_MODEL == "google/flan-t5-large"
        assert settings.RAG_GENERATION_MODEL_LIGHT == "google/flan-t5-base"

    def test_get_generation_model_name_quality(self):
        """get_generation_model_name should return large model for quality tier."""
        model = get_generation_model_name()
        assert model == "google/flan-t5-large"

    def test_get_generation_model_name_speed(self, monkeypatch):
        """get_generation_model_name should return base model for speed tier."""
        monkeypatch.setenv("RAG_GENERATION_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        model = get_generation_model_name()
        assert model == "google/flan-t5-base"

    def test_get_summarization_model_name_quality(self):
        """get_summarization_model_name should return BART-large for quality tier."""
        # Ensure clean settings state
        from backend.config import reset_settings
        reset_settings()
        import os
        os.environ["RAG_GENERATION_TIER"] = "quality"
        
        model = get_summarization_model_name()
        assert model == "facebook/bart-large-cnn"

    def test_get_summarization_model_name_speed(self, monkeypatch):
        """get_summarization_model_name should return distilbart for speed tier."""
        monkeypatch.setenv("RAG_GENERATION_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        model = get_summarization_model_name()
        assert model == "sshleifer/distilbart-cnn-12-6"


class TestTierIndependence:
    """Test that embedding and generation tiers are independent."""

    def test_embedding_tier_change_does_not_affect_generation(self, monkeypatch):
        """Changing embedding tier should not affect generation model."""
        # Set embedding to speed, generation to quality
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "speed")
        monkeypatch.setenv("RAG_GENERATION_TIER", "quality")
        from backend.config import reset_settings
        reset_settings()
        
        emb_model = get_embedding_model_name()
        gen_model = get_generation_model_name()
        
        assert "bge-small" in emb_model
        assert gen_model == "google/flan-t5-large"

    def test_generation_tier_change_does_not_affect_embedding(self, monkeypatch):
        """Changing generation tier should not affect embedding model."""
        # Set embedding to quality, generation to speed
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "quality")
        monkeypatch.setenv("RAG_GENERATION_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        emb_model = get_embedding_model_name()
        gen_model = get_generation_model_name()
        
        assert "bge-base" in emb_model
        assert gen_model == "google/flan-t5-base"

    def test_both_speed_tiers(self, monkeypatch):
        """Both tiers can be set to speed independently."""
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "speed")
        monkeypatch.setenv("RAG_GENERATION_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        emb_model = get_embedding_model_name()
        gen_model = get_generation_model_name()
        
        assert "bge-small" in emb_model
        assert gen_model == "google/flan-t5-base"

    def test_both_quality_tiers(self, monkeypatch):
        """Both tiers can be set to quality (default)."""
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "quality")
        monkeypatch.setenv("RAG_GENERATION_TIER", "quality")
        from backend.config import reset_settings
        reset_settings()
        
        emb_model = get_embedding_model_name()
        gen_model = get_generation_model_name()
        
        assert "bge-base" in emb_model
        assert gen_model == "google/flan-t5-large"


class TestEmbeddingConfigFromSettings:
    """Test EmbeddingConfig creation from settings."""

    def test_create_embedding_config_quality(self):
        """create_embedding_config_from_settings should use quality model by default."""
        config = create_embedding_config_from_settings()
        assert config.model_name == "BAAI/bge-base-en-v1.5"
        assert config.expected_dimension == 768

    def test_create_embedding_config_speed(self, monkeypatch):
        """create_embedding_config_from_settings should use small model for speed tier."""
        monkeypatch.setenv("RAG_EMBEDDING_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        config = create_embedding_config_from_settings()
        assert config.model_name == "BAAI/bge-small-en-v1.5"
        assert config.expected_dimension == 384


class TestVectorStoreConfigEmbeddingMetadata:
    """Test VectorStoreConfig includes embedding metadata."""

    def test_vector_store_config_has_embedding_fields(self):
        """VectorStoreConfig should have embedding dimension and model fields."""
        config = VectorStoreConfig(
            expected_embedding_dimension=768,
            embedding_model_name="BAAI/bge-base-en-v1.5"
        )
        assert config.expected_embedding_dimension == 768
        assert config.embedding_model_name == "BAAI/bge-base-en-v1.5"

    def test_vector_store_config_defaults(self):
        """VectorStoreConfig should have sensible defaults."""
        config = VectorStoreConfig()
        assert config.expected_embedding_dimension == 768
        assert config.embedding_model_name == "BAAI/bge-base-en-v1.5"


class TestBackwardCompatibility:
    """Test backward compatibility with existing environment variables."""

    def test_model_tier_legacy_still_works(self, monkeypatch):
        """Legacy MODEL_TIER should still work for generation."""
        monkeypatch.setenv("MODEL_TIER", "speed")
        from backend.config import reset_settings
        reset_settings()
        
        settings = get_settings()
        # MODEL_TIER should affect generation tier for backward compatibility
        # but NOT embedding tier
        assert settings.MODEL_TIER == "speed"

    def test_sbert_model_alias(self, monkeypatch):
        """SBERT_MODEL alias should work."""
        monkeypatch.setenv("SBERT_MODEL", "BAAI/bge-small-en-v1.5")
        from backend.config import reset_settings
        reset_settings()
        
        settings = get_settings()
        assert settings.SBERT_MODEL == "BAAI/bge-small-en-v1.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])