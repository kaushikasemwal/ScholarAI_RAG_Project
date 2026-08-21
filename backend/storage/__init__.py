"""
storage/__init__.py — Storage Abstraction Layer
================================================
Unified interface for local, S3, and GCS storage backends.
"""

from .base import StorageBackend, StorageConfig
from .factory import create_storage, get_storage
from .gcs import GCSStorage
from .local import LocalStorage
from .s3 import S3Storage

__all__ = [
    "StorageBackend",
    "StorageConfig",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "get_storage",
    "create_storage",
]
