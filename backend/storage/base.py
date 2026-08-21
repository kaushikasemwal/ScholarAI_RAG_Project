"""
storage/base.py — Storage Backend Interface
============================================
Abstract base class for storage backends.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass
class StorageConfig:
    """Base configuration for storage backends."""
    pass


class ObjectMetadata(BaseModel):
    """Metadata for a stored object."""
    key: str = Field(..., description="Object key/path")
    size_bytes: int = Field(..., description="Object size in bytes")
    content_type: str | None = Field(None, description="MIME type")
    last_modified: datetime = Field(..., description="Last modification time")
    etag: str | None = Field(None, description="Entity tag for versioning")
    metadata: dict = Field(default_factory=dict, description="Custom metadata")


class PresignedUrlRequest(BaseModel):
    """Request for generating a presigned URL."""
    key: str = Field(..., description="Object key")
    expiration_seconds: int = Field(default=3600, description="URL expiration time")
    method: str = Field(default="GET", description="HTTP method (GET, PUT, etc.)")
    content_type: str | None = Field(None, description="Content type for PUT")


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        """
        Upload an object.
        
        Args:
            key: Object key (path)
            data: Object data
            content_type: MIME type
            metadata: Custom metadata
            
        Returns:
            Object metadata
        """
        pass

    @abstractmethod
    async def put_object_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        """Upload an object from an async stream."""
        pass

    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """
        Download an object.
        
        Args:
            key: Object key
            
        Returns:
            Object data
            
        Raises:
            KeyError: If object doesn't exist
        """
        pass

    @abstractmethod
    async def get_object_stream(self, key: str) -> AsyncIterator[bytes]:
        """Download an object as an async stream."""
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """
        Delete an object.
        
        Args:
            key: Object key
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def head_object(self, key: str) -> ObjectMetadata:
        """
        Get object metadata without downloading.
        
        Args:
            key: Object key
            
        Returns:
            Object metadata
            
        Raises:
            KeyError: If object doesn't exist
        """
        pass

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
    ) -> list[ObjectMetadata]:
        """
        List objects with optional prefix filtering.
        
        Args:
            prefix: Key prefix filter
            delimiter: Delimiter for hierarchical listing
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata
        """
        pass

    @abstractmethod
    async def generate_presigned_url(self, request: PresignedUrlRequest) -> str:
        """
        Generate a presigned URL for direct object access.
        
        Args:
            request: Presigned URL request
            
        Returns:
            Signed URL string
        """
        pass

    @abstractmethod
    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        """Copy an object within the same backend."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass

    @abstractmethod
    def get_public_url(self, key: str) -> str:
        """Get public URL for an object (if applicable)."""
        pass
