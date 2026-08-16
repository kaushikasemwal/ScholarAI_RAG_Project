"""
storage/factory.py — Storage Factory
=====================================
Factory for creating storage backends from configuration.
"""

from typing import Optional

from .base import StorageBackend
from .local import LocalStorage
from .s3 import S3Storage
from .gcs import GCSStorage
from ..config import get_settings


_storage_instance: Optional[StorageBackend] = None


def create_storage(backend: Optional[str] = None) -> StorageBackend:
    """
    Create a storage backend instance.
    
    Args:
        backend: Backend type ('local', 's3', 'gcs'). 
                 If None, reads from settings.STORAGE_BACKEND.
                 
    Returns:
        Configured StorageBackend instance.
    """
    settings = get_settings()
    
    backend = backend or getattr(settings, "STORAGE_BACKEND", "local")
    
    if backend == "local":
        path = getattr(settings, "STORAGE_LOCAL_PATH", "storage")
        return LocalStorage(base_path=path)
    
    elif backend == "s3":
        return S3Storage(
            bucket=settings.STORAGE_S3_BUCKET,
            region=getattr(settings, "STORAGE_S3_REGION", "us-east-1"),
            endpoint_url=getattr(settings, "STORAGE_S3_ENDPOINT_URL", None),
            access_key_id=getattr(settings, "STORAGE_S3_ACCESS_KEY", None),
            secret_access_key=getattr(settings, "STORAGE_S3_SECRET_KEY", None),
            prefix=getattr(settings, "STORAGE_S3_PREFIX", ""),
        )
    
    elif backend == "gcs":
        return GCSStorage(
            bucket=settings.STORAGE_GCS_BUCKET,
            project=getattr(settings, "STORAGE_GCS_PROJECT", None),
            credentials_path=getattr(settings, "STORAGE_GCS_CREDENTIALS", None),
            prefix=getattr(settings, "STORAGE_GCS_PREFIX", ""),
        )
    
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


def get_storage() -> StorageBackend:
    """Get the global storage instance (singleton)."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = create_storage()
    return _storage_instance


def set_storage(backend: StorageBackend):
    """Set the global storage instance."""
    global _storage_instance
    _storage_instance = backend