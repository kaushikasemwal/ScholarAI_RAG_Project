"""
config.py — Centralized Configuration Management
=================================================
Pydantic Settings for type-safe, environment-driven configuration.
"""

import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────────
    APP_NAME: str = "ScholarAI API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="development|staging|production")
    DEBUG: bool = False

    # ─── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ─── CORS ─────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] | str = Field(
        default="*",
        description="Comma-separated list of allowed origins or JSON array"
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["*"]
            # Try JSON array first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # Fallback: comma-separated
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return ["*"]

    # ─── Rate Limiting ────────────────────────────────────────────
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_CLEANUP: str = "30/minute"

    # ─── Redis ─────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_ENABLED: bool = True

    # ─── File Storage ─────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")
    MODELS_DIR: Path = Path("models")
    MAX_UPLOAD_SIZE_MB: int = 50
    AUTO_DELETE_HOURS: int = 1

    # ─── Encryption ───────────────────────────────────────────────
    AES_KEY: str | None = Field(
        default=None,
        description="Base64-encoded 32-byte AES key (optional, auto-generated if not set)"
    )
    MAX_KEY_VERSIONS: int = 5

    # ─── Logging ──────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    LOG_SERVICE_NAME: str = "scholarai-api"

    # ─── Generation Workers ───────────────────────────────────────
    GENERATION_WORKERS: int = 2
    GENERATION_TIMEOUT_SECONDS: int = 300

    # ─── ML Models ────────────────────────────────────────────────
    # Model names (can override for different variants)
    # Embeddings: BGE-base (768-dim) for better quality, or bge-small (384-dim) for speed
    RAG_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RAG_EMBEDDING_DIMENSION: int = 768
    # Summarization: BART-large-CNN (stronger than Pegasus-xsum for general docs)
    PEGASUS_MODEL: str = "facebook/bart-large-cnn"
    # Quiz Generation: FLAN-T5-large (instruction-tuned, better QG than base T5)
    RAG_GENERATION_MODEL: str = "google/flan-t5-large"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Alternative lighter models (for CPU/memory constrained environments)
    RAG_EMBEDDING_MODEL_LIGHT: str = "BAAI/bge-small-en-v1.5"
    RAG_EMBEDDING_DIMENSION_LIGHT: int = 384
    PEGASUS_MODEL_LIGHT: str = "sshleifer/distilbart-cnn-12-6"
    RAG_GENERATION_MODEL_LIGHT: str = "google/flan-t5-base"
    
    # Embedding tier selection: "quality" | "speed"
    RAG_EMBEDDING_TIER: str = "quality"
    # Generation tier selection: "quality" | "speed"
    RAG_GENERATION_TIER: str = "quality"
    
    # Model selection: "quality" | "balanced" | "speed" (legacy, deprecated)
    # Controls generation tier only; embedding tier uses RAG_EMBEDDING_TIER
    MODEL_TIER: str = "quality"
    
    # Backward compatibility aliases (deprecated, will be removed)
    SBERT_MODEL: str = "BAAI/bge-base-en-v1.5"
    SBERT_MODEL_LIGHT: str = "BAAI/bge-small-en-v1.5"
    PEGASUS_MODEL: str = "facebook/bart-large-cnn"
    PEGASUS_MODEL_LIGHT: str = "sshleifer/distilbart-cnn-12-6"
    T5_MODEL: str = "google/flan-t5-large"
    T5_MODEL_LIGHT: str = "google/flan-t5-base"
    T5_MODEL: str = "google/flan-t5-large"
    T5_MODEL_LIGHT: str = "google/flan-t5-base"
    SBERT_MODEL: str = "BAAI/bge-base-en-v1.5"
    SBERT_MODEL_LIGHT: str = "BAAI/bge-small-en-v1.5"
    
    # Embedding model -> dimension mapping (for validation)
    # Centralized to avoid hardcoding in multiple places
    EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "all-MiniLM-L6-v2": 384,
    }

    # Autoencoder settings
    AUTOENCODER_INPUT_DIM: int = 768  # Matches BGE-base
    AUTOENCODER_LATENT_DIM: int = 256
    AUTOENCODER_EPOCHS: int = 50
    AUTOENCODER_LR: float = 1e-3
    AUTOENCODER_BATCH_SIZE: int = 64

    # Summarization settings
    SUMMARY_MAX_SENTENCES: int = 80
    SUMMARY_TOP_K: int = 15
    SUMMARY_MAX_LENGTH: int = 256
    SUMMARY_MIN_LENGTH: int = 80

    # Quiz settings
    QUIZ_QUESTIONS: int = 10
    QUIZ_CHUNK_SIZE: int = 400
    QUIZ_MAX_CHUNKS: int = 20

    # TTS settings
    TTS_CHUNK_SIZE: int = 500
    TTS_MIN_AUDIO_BYTES: int = 500

    # Video settings
    VIDEO_WIDTH: int = 640
    VIDEO_HEIGHT: int = 360
    VIDEO_FPS: int = 10
    VIDEO_WORDS_PER_MINUTE: int = 150
    VIDEO_MIN_SLIDE_SECONDS: int = 5
    VIDEO_MAX_SLIDE_SECONDS: int = 20

    # ─── RAG Settings (Phase 1) ─────────────────────────────────────
    RAG_ENABLED: bool = True
    RAG_VECTOR_STORE: str = "chroma"  # chroma, vectorize (future)
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 100
    RAG_SIMILARITY_THRESHOLD: float | None = None
    RAG_VECTOR_STORE_DIR: str = "./chroma_db"
    RAG_COLLECTION_NAME: str = "scholarai_documents"
    RAG_DISTANCE_METRIC: str = "cosine"

    # ─── RAG Settings (Phase 3) ──────────────────────────────────────
    RAG_RETRIEVAL_MIN_SCORE: float = 0.5  # Minimum cosine similarity for retrieval
    RAG_MAX_RETRIES: int = 3  # Max retries per question
    RAG_ANSWER_MIN_SUPPORT_SCORE: float = 0.6  # Minimum semantic match for answer validation
    RAG_DISTRACTOR_MIN_SIMILARITY: float = 0.2  # Minimum similarity for distractors
    RAG_DISTRACTOR_MAX_SIMILARITY: float = 0.7  # Maximum similarity for distractors
    RAG_DISTRACTOR_MAX_OVERLAP: float = 0.5  # Max word overlap with correct answer
    RAG_QUESTION_MIN_LENGTH: int = 10  # Minimum question length (words)
    RAG_QUESTION_MIN_LENGTH_ANSWER_FIRST: int = 6  # Minimum question length for answer-first approach
    RAG_REJECT_GENERIC: bool = True  # Reject generic/template questions
    RAG_REJECT_ARTIFACTS: bool = True  # Reject questions with PDF artifacts

    # ─── Firebase (Frontend) ──────────────────────────────────────
    FIREBASE_API_KEY: str | None = None
    FIREBASE_AUTH_DOMAIN: str | None = None
    FIREBASE_PROJECT_ID: str | None = None
    FIREBASE_STORAGE_BUCKET: str | None = None
    FIREBASE_MESSAGING_SENDER_ID: str | None = None
    FIREBASE_APP_ID: str | None = None

    # ─── Firebase Admin (Backend) ───────────────────────────────────
    FIREBASE_SERVICE_ACCOUNT_JSON: str | None = Field(
        default=None,
        description="Firebase service account JSON for Admin SDK (base64 or raw JSON)"
    )
    GOOGLE_APPLICATION_CREDENTIALS: str | None = Field(
        default=None,
        description="Path to Google Cloud service account key file"
    )

    # ─── Hugging Face / Deployment ────────────────────────────────
    HF_TOKEN: str | None = None
    HF_SPACE_ID: str | None = None
    HF_SPACE_URL: str | None = None

    # ─── Azure / Jenkins ──────────────────────────────────────────
    AZURE_SUBSCRIPTION_ID: str | None = None
    AZURE_TENANT_ID: str | None = None
    AZURE_CLIENT_ID: str | None = None
    AZURE_CLIENT_SECRET: str | None = None
    ACR_NAME: str = "scholarairegistry"
    ACR_LOGIN_SERVER: str = "scholarairegistry.azurecr.io"
    RESOURCE_GROUP: str = "ScholarAI-RG"
    CONTAINER_APP_NAME: str = "scholarai-backend"

    # ─── Storage ───────────────────────────────────────────────────
    STORAGE_BACKEND: str = "local"  # local, s3, gcs
    STORAGE_LOCAL_PATH: str = "storage"

    # S3 settings
    STORAGE_S3_BUCKET: str | None = None
    STORAGE_S3_REGION: str = "us-east-1"
    STORAGE_S3_ENDPOINT_URL: str | None = None
    STORAGE_S3_ACCESS_KEY: str | None = None
    STORAGE_S3_SECRET_KEY: str | None = None
    STORAGE_S3_PREFIX: str = ""

    # GCS settings
    STORAGE_GCS_BUCKET: str | None = None
    STORAGE_GCS_PROJECT: str | None = None
    STORAGE_GCS_CREDENTIALS: str | None = None
    STORAGE_GCS_PREFIX: str = ""


# ─── HELPER FUNCTIONS ────────────────────────────────────────────


def get_embedding_dimension(model_name: str | None = None) -> int:
    """
    Get the expected embedding dimension for a model.
    
    Args:
        model_name: Embedding model name. If None, uses RAG_EMBEDDING_MODEL from settings.
        
    Returns:
        Expected embedding dimension (768 for BGE-base, 384 for BGE-small).
    """
    settings = get_settings()
    model = model_name or settings.RAG_EMBEDDING_MODEL
    return settings.EMBEDDING_MODEL_DIMENSIONS.get(model, 768)


def get_generation_model_name() -> str:
    """
    Get the configured generation model name based on RAG_GENERATION_TIER.
    
    Returns:
        Model name for quiz generation (FLAN-T5).
    """
    settings = get_settings()
    if settings.RAG_GENERATION_TIER == "speed":
        return settings.RAG_GENERATION_MODEL_LIGHT
    return settings.RAG_GENERATION_MODEL


def get_summarization_model_name() -> str:
    """
    Get the configured summarization model name based on RAG_GENERATION_TIER.
    
    Returns:
        Model name for summarization (BART).
    """
    settings = get_settings()
    if settings.RAG_GENERATION_TIER == "speed":
        return settings.PEGASUS_MODEL_LIGHT
    return settings.PEGASUS_MODEL


def get_embedding_model_name() -> str:
    """
    Get the configured embedding model name based on RAG_EMBEDDING_TIER.
    
    Returns:
        Model name for embeddings (BGE).
    """
    settings = get_settings()
    if settings.RAG_EMBEDDING_TIER == "speed":
        return settings.RAG_EMBEDDING_MODEL_LIGHT
    return settings.RAG_EMBEDDING_MODEL


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None