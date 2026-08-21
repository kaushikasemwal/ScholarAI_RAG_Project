"""
storage/gcs.py — Google Cloud Storage
======================================
GCS implementation of StorageBackend using google-cloud-storage.
"""

import mimetypes
from collections.abc import AsyncIterator
from datetime import datetime

from google.cloud import storage
from google.cloud.storage import Blob

from .base import ObjectMetadata, PresignedUrlRequest, StorageBackend


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket: str,
        project: str | None = None,
        credentials_path: str | None = None,
        prefix: str = "",
    ):
        self.bucket_name = bucket
        self.project = project
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        if credentials_path:
            self.client = storage.Client.from_service_account_json(credentials_path, project=project)
        else:
            self.client = storage.Client(project=project)

        self.bucket = self.client.bucket(bucket)

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key.lstrip('/')}"

    def _strip_prefix(self, key: str) -> str:
        if key.startswith(self.prefix):
            return key[len(self.prefix):]
        return key

    def _get_mime_type(self, key: str) -> str:
        mime, _ = mimetypes.guess_type(key)
        return mime or "application/octet-stream"

    def _blob_to_metadata(self, blob: Blob, key: str) -> ObjectMetadata:
        return ObjectMetadata(
            key=self._strip_prefix(key),
            size_bytes=blob.size or 0,
            content_type=blob.content_type or self._get_mime_type(key),
            last_modified=blob.updated or datetime.utcnow(),
            etag=blob.etag.strip('"') if blob.etag else None,
            metadata=blob.metadata or {},
        )

    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)

        if content_type:
            blob.content_type = content_type
        if metadata:
            blob.metadata = metadata

        # Upload from bytes
        blob.upload_from_string(data, content_type=content_type)

        return await self.head_object(key)

    async def put_object_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        data = b"".join(chunks)
        return await self.put_object(key, data, content_type, metadata)

    async def get_object(self, key: str) -> bytes:
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)

        if not blob.exists():
            raise KeyError(f"Object not found: {key}")

        return blob.download_as_bytes()

    async def get_object_stream(self, key: str) -> AsyncIterator[bytes]:
        # GCS doesn't have native streaming download, so we chunk
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)

        if not blob.exists():
            raise KeyError(f"Object not found: {key}")

        # Download in chunks using byte ranges
        chunk_size = 64 * 1024  # 64KB
        total_size = blob.size
        offset = 0

        while offset < total_size:
            end = min(offset + chunk_size - 1, total_size - 1)
            chunk_data = blob.download_as_bytes(start=offset, end=end)
            yield chunk_data
            offset = end + 1

    async def delete_object(self, key: str) -> bool:
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)

        if blob.exists():
            blob.delete()
            return True
        return False

    async def head_object(self, key: str) -> ObjectMetadata:
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)

        if not blob.exists():
            raise KeyError(f"Object not found: {key}")

        blob.reload()
        return self._blob_to_metadata(blob, full_key)

    async def list_objects(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
    ) -> list[ObjectMetadata]:
        full_prefix = self._make_key(prefix)

        blobs = self.client.list_blobs(
            self.bucket_name,
            prefix=full_prefix,
            delimiter=delimiter,
            max_results=max_keys,
        )

        results = []
        for blob in blobs:
            if not blob.name.endswith("/"):  # Skip directory markers
                results.append(self._blob_to_metadata(blob, blob.name))

        return results

    async def generate_presigned_url(self, request: PresignedUrlRequest) -> str:
        full_key = self._make_key(request.key)
        blob = self.bucket.blob(full_key)

        if not blob.exists():
            raise KeyError(f"Object not found: {request.key}")

        expiration = datetime.utcnow() + timedelta(seconds=request.expiration_seconds)

        if request.method.upper() == "GET":
            return blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET",
                content_type=request.content_type,
            )
        elif request.method.upper() == "PUT":
            return blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="PUT",
                content_type=request.content_type,
            )
        else:
            raise ValueError(f"Unsupported method: {request.method}")

    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        source_full = self._make_key(source_key)
        dest_full = self._make_key(dest_key)

        source_blob = self.bucket.blob(source_full)
        dest_blob = self.bucket.blob(dest_full)

        if not source_blob.exists():
            raise KeyError(f"Source object not found: {source_key}")

        self.bucket.copy_blob(source_blob, self.bucket, dest_full)

        return await self.head_object(dest_key)

    async def exists(self, key: str) -> bool:
        full_key = self._make_key(key)
        blob = self.bucket.blob(full_key)
        return blob.exists()

    def get_public_url(self, key: str) -> str:
        full_key = self._make_key(key)
        return f"https://storage.googleapis.com/{self.bucket_name}/{full_key}"
