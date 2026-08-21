"""
storage/s3.py — AWS S3 Storage
================================
AWS S3 implementation of StorageBackend using aiobotocore.
"""

import mimetypes
from collections.abc import AsyncIterator
from datetime import datetime

from aiobotocore.session import get_session

from .base import ObjectMetadata, PresignedUrlRequest, StorageBackend


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        prefix: str = "",
    ):
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        self._session = get_session()
        self._client = None

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            self._client = await self._session.create_client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            ).__aenter__()
        return self._client

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key.lstrip('/')}"

    def _strip_prefix(self, key: str) -> str:
        """Remove prefix from key."""
        if key.startswith(self.prefix):
            return key[len(self.prefix):]
        return key

    def _get_mime_type(self, key: str) -> str:
        mime, _ = mimetypes.guess_type(key)
        return mime or "application/octet-stream"

    def _s3obj_to_metadata(self, obj: dict, key: str) -> ObjectMetadata:
        return ObjectMetadata(
            key=self._strip_prefix(key),
            size_bytes=obj.get("Size", 0),
            content_type=obj.get("ContentType") or self._get_mime_type(key),
            last_modified=obj.get("LastModified", datetime.utcnow()),
            etag=obj.get("ETag", "").strip('"'),
        )

    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        client = await self._get_client()
        full_key = self._make_key(key)

        kwargs = {
            "Bucket": self.bucket,
            "Key": full_key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        if metadata:
            kwargs["Metadata"] = metadata

        await client.put_object(**kwargs)

        # Head to get metadata
        return await self.head_object(key)

    async def put_object_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> ObjectMetadata:
        # For streaming, we need to collect into bytes first
        # In production, use multipart upload for large streams
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        data = b"".join(chunks)
        return await self.put_object(key, data, content_type, metadata)

    async def get_object(self, key: str) -> bytes:
        client = await self._get_client()
        full_key = self._make_key(key)

        try:
            response = await client.get_object(Bucket=self.bucket, Key=full_key)
            async with response["Body"] as stream:
                return await stream.read()
        except client.exceptions.NoSuchKey:
            raise KeyError(f"Object not found: {key}")

    async def get_object_stream(self, key: str) -> AsyncIterator[bytes]:
        client = await self._get_client()
        full_key = self._make_key(key)

        try:
            response = await client.get_object(Bucket=self.bucket, Key=full_key)
            async with response["Body"] as stream:
                async for chunk in stream.iter_chunks():
                    yield chunk
        except client.exceptions.NoSuchKey:
            raise KeyError(f"Object not found: {key}")

    async def delete_object(self, key: str) -> bool:
        client = await self._get_client()
        full_key = self._make_key(key)

        try:
            await client.delete_object(Bucket=self.bucket, Key=full_key)
            return True
        except client.exceptions.NoSuchKey:
            return False

    async def head_object(self, key: str) -> ObjectMetadata:
        client = await self._get_client()
        full_key = self._make_key(key)

        try:
            response = await client.head_object(Bucket=self.bucket, Key=full_key)
            return ObjectMetadata(
                key=key,
                size_bytes=response.get("ContentLength", 0),
                content_type=response.get("ContentType") or self._get_mime_type(key),
                last_modified=response.get("LastModified", datetime.utcnow()),
                etag=response.get("ETag", "").strip('"'),
                metadata=response.get("Metadata", {}),
            )
        except client.exceptions.NoSuchKey:
            raise KeyError(f"Object not found: {key}")

    async def list_objects(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
    ) -> list[ObjectMetadata]:
        client = await self._get_client()
        full_prefix = self._make_key(prefix)

        kwargs = {
            "Bucket": self.bucket,
            "Prefix": full_prefix,
            "MaxKeys": max_keys,
        }
        if delimiter:
            kwargs["Delimiter"] = delimiter

        response = await client.list_objects_v2(**kwargs)

        results = []
        for obj in response.get("Contents", []):
            results.append(self._s3obj_to_metadata(obj, obj["Key"]))

        return results

    async def generate_presigned_url(self, request: PresignedUrlRequest) -> str:
        client = await self._get_client()
        full_key = self._make_key(request.key)

        if request.method.upper() == "GET":
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": full_key},
                ExpiresIn=request.expiration_seconds,
            )
        elif request.method.upper() == "PUT":
            params = {"Bucket": self.bucket, "Key": full_key}
            if request.content_type:
                params["ContentType"] = request.content_type
            url = await client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=request.expiration_seconds,
            )
        else:
            raise ValueError(f"Unsupported method: {request.method}")

        return url

    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        client = await self._get_client()
        source_full = self._make_key(source_key)
        dest_full = self._make_key(dest_key)

        await client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": source_full},
            Key=dest_full,
        )

        return await self.head_object(dest_key)

    async def exists(self, key: str) -> bool:
        try:
            await self.head_object(key)
            return True
        except KeyError:
            return False

    def get_public_url(self, key: str) -> str:
        full_key = self._make_key(key)
        if self.endpoint_url:
            return f"{self.endpoint_url}/{self.bucket}/{full_key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{full_key}"

    async def close(self):
        """Close the S3 client."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
