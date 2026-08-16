"""
config.py — Centralized Configuration Management
=================================================
Pydantic Settings for type-safe, environment-driven configuration.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


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
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        description="Comma-separated list of allowed origins"
    )
    
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
    AES_KEY: Optional[str] = Field(
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
    SBERT_MODEL: str = "BAAI/bge-small-en-v1.5"
    PEGASUS_MODEL: str = "google/pegasus-xsum"
    T5_MODEL: str = "valhalla/t5-base-qg-hl"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Autoencoder settings
    AUTOENCODER_INPUT_DIM: int = 384
    AUTOENCODER_LATENT_DIM: int = 128
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
    
    # ─── Firebase (Frontend) ──────────────────────────────────────
    FIREBASE_API_KEY: Optional[str] = None
    FIREBASE_AUTH_DOMAIN: Optional[str] = None
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None
    FIREBASE_MESSAGING_SENDER_ID: Optional[str] = None
    FIREBASE_APP_ID: Optional[str] = None
    
    # ─── Firebase Admin (Backend) ───────────────────────────────────
    FIREBASE_SERVICE_ACCOUNT_JSON: Optional[str] = Field(
        default=None,
        description="Firebase service account JSON for Admin SDK (base64 or raw JSON)"
    )
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud service account key file"
    )
    
    # ─── Hugging Face / Deployment ────────────────────────────────
    HF_TOKEN: Optional[str] = None
    HF_SPACE_ID: Optional[str] = None
    HF_SPACE_URL: Optional[str] = None
    
    # ─── Azure / Jenkins ──────────────────────────────────────────
    AZURE_SUBSCRIPTION_ID: Optional[str] = None
    AZURE_TENANT_ID: Optional[str] = None
    AZURE_CLIENT_ID: Optional[str] = None
    AZURE_CLIENT_SECRET: Optional[str] = None
    ACR_NAME: str = "scholarairegistry"
    ACR_LOGIN_SERVER: str = "scholarairegistry.azurecr.io"
    RESOURCE_GROUP: str = "ScholarAI-RG"
    CONTAINER_APP_NAME: str = "scholarai-backend"

    # ─── Storage ───────────────────────────────────────────────────
    STORAGE_BACKEND: str = "local"  # local, s3, gcs
    STORAGE_LOCAL_PATH: str = "storage"
    
    # S3 settings
    STORAGE_S3_BUCKET: Optional[str] = None
    STORAGE_S3_REGION: str = "us-east-1"
    STORAGE_S3_ENDPOINT_URL: Optional[str] = None
    STORAGE_S3_ACCESS_KEY: Optional[str] = None
    STORAGE_S3_SECRET_KEY: Optional[str] = None
    STORAGE_S3_PREFIX: str = ""
    
    # GCS settings
    STORAGE_GCS_BUCKET: Optional[str] = None
    STORAGE_GCS_PROJECT: Optional[str] = None
    STORAGE_GCS_CREDENTIALS: Optional[str] = None
    STORAGE_GCS_PREFIX: str = ""

# Global settings instance
_settings: Optional[Settings] = None


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