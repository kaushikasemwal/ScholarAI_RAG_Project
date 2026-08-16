"""
storage/__init__.py — Storage Abstraction Layer
================================================
Unified interface for local, S3, and GCS storage backends.
"""

from .base import StorageBackend, StorageConfig
from .local import LocalStorage
from .s3 import S3Storage
from .gcs import GCSStorage
from .factory import get_storage, create_storage

__all__ = [
    "StorageBackend",
    "StorageConfig", 
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "get_storage",
    "create_storage",
]