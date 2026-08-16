"""
storage/local.py — Local Filesystem Storage
============================================
Local filesystem implementation of StorageBackend.
"""

import aiofiles
import aiofiles.os
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional

from .base import ObjectMetadata, PresignedUrlRequest, StorageBackend


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, key: str) -> Path:
        """Get full filesystem path for a key."""
        # Sanitize key to prevent directory traversal
        safe_key = key.lstrip("/").replace("..", "")
        return self.base_path / safe_key
    
    def _get_mime_type(self, key: str) -> str:
        """Guess MIME type from file extension."""
        mime, _ = mimetypes.guess_type(key)
        return mime or "application/octet-stream"
    
    def _stat_to_metadata(self, key: str, stat) -> ObjectMetadata:
        """Convert os.stat result to ObjectMetadata."""
        return ObjectMetadata(
            key=key,
            size_bytes=stat.st_size,
            content_type=self._get_mime_type(key),
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            etag=f'"{stat.st_mtime_ns}-{stat.st_size}"',
        )
    
    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ObjectMetadata:
        path = self._get_full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        
        stat = await aiofiles.os.stat(path)
        meta = self._stat_to_metadata(key, stat)
        if content_type:
            meta.content_type = content_type
        if metadata:
            meta.metadata = metadata
        return meta
    
    async def put_object_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ObjectMetadata:
        path = self._get_full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, "wb") as f:
            async for chunk in stream:
                await f.write(chunk)
        
        stat = await aiofiles.os.stat(path)
        meta = self._stat_to_metadata(key, stat)
        if content_type:
            meta.content_type = content_type
        if metadata:
            meta.metadata = metadata
        return meta
    
    async def get_object(self, key: str) -> bytes:
        path = self._get_full_path(key)
        if not await aiofiles.os.path.exists(path):
            raise KeyError(f"Object not found: {key}")
        
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    
    async def get_object_stream(self, key: str) -> AsyncIterator[bytes]:
        path = self._get_full_path(key)
        if not await aiofiles.os.path.exists(path):
            raise KeyError(f"Object not found: {key}")
        
        async with aiofiles.open(path, "rb") as f:
            chunk_size = 64 * 1024  # 64KB chunks
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async def delete_object(self, key: str) -> bool:
        path = self._get_full_path(key)
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)
            return True
        return False
    
    async def head_object(self, key: str) -> ObjectMetadata:
        path = self._get_full_path(key)
        if not await aiofiles.os.path.exists(path):
            raise KeyError(f"Object not found: {key}")
        
        stat = await aiofiles.os.stat(path)
        return self._stat_to_metadata(key, stat)
    
    async def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[ObjectMetadata]:
        prefix_path = self._get_full_path(prefix)
        if not await aiofiles.os.path.exists(prefix_path):
            return []
        
        results = []
        if await aiofiles.os.path.isfile(prefix_path):
            stat = await aiofiles.os.stat(prefix_path)
            results.append(self._stat_to_metadata(prefix, stat))
        else:
            async for entry in aiofiles.os.scandir(prefix_path):
                if entry.is_file():
                    rel_key = str(Path(entry.path).relative_to(self.base_path))
                    stat = await aiofiles.os.stat(entry.path)
                    results.append(self._stat_to_metadata(rel_key, stat))
                elif entry.is_dir() and delimiter is None:
                    # Recurse into subdirectories if no delimiter
                    sub_results = await self.list_objects(
                        prefix=str(Path(prefix) / entry.name),
                        delimiter=delimiter,
                        max_keys=max_keys - len(results),
                    )
                    results.extend(sub_results)
                
                if len(results) >= max_keys:
                    break
        
        return results[:max_keys]
    
    async def generate_presigned_url(self, request: PresignedUrlRequest) -> str:
        # Local storage: return file:// URL or local server URL
        path = self._get_full_path(request.key)
        if await aiofiles.os.path.exists(path):
            return f"file://{path}"
        raise KeyError(f"Object not found: {request.key}")
    
    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        source_path = self._get_full_path(source_key)
        dest_path = self._get_full_path(dest_key)
        
        if not await aiofiles.os.path.exists(source_path):
            raise KeyError(f"Source object not found: {source_key}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy in chunks
        async with aiofiles.open(source_path, "rb") as src:
            async with aiofiles.open(dest_path, "wb") as dst:
                chunk_size = 64 * 1024
                while True:
                    chunk = await src.read(chunk_size)
                    if not chunk:
                        break
                    await dst.write(chunk)
        
        stat = await aiofiles.os.stat(dest_path)
        return self._stat_to_metadata(dest_key, stat)
    
    async def exists(self, key: str) -> bool:
        path = self._get_full_path(key)
        return await aiofiles.os.path.exists(path)
    
    def get_public_url(self, key: str) -> str:
        return f"file://{self._get_full_path(key)}"