"""Storage abstraction for scan images.

Two implementations:

* ``LocalFsStorage`` — writes under ``settings.storage_local_path``. Used
  in dev and tests.
* ``S3Storage`` — uses boto3, with presigned URLs for direct mobile
  uploads.

The Protocol intentionally exposes only the four methods the API needs:
``put`` (server-side write of bytes), ``get`` (read), ``delete``, and
``generate_signed_url`` for the mobile-direct-upload path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...
    async def generate_signed_url(
        self,
        key: str,
        *,
        surface: str | None = None,
        expires_in: int = 900,
        method: str = "PUT",
    ) -> str: ...


def scan_image_key(scan_id: str, surface: str) -> str:
    """Conventional storage key for a scan's surface image.

    Centralized so both server-side puts and presign flows agree.
    """
    return f"scans/{scan_id}/{surface}"


class LocalFsStorage:
    """Writes blobs to a directory tree under ``base_path``.

    Keys map 1:1 to relative paths. Used in tests and local dev. Signed
    URLs return a path-like string that the API can route back to the
    PUT upload endpoint — there is no real signing.
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Defense in depth: don't allow key escape.
        candidate = (self.base_path / key).resolve()
        if not str(candidate).startswith(str(self.base_path.resolve())):
            raise ValueError(f"key {key!r} escapes base path")
        return candidate

    async def put(self, key: str, data: bytes) -> None:
        def _write() -> None:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

        await asyncio.to_thread(_write)

    async def get(self, key: str) -> bytes:
        def _read() -> bytes:
            return self._path(key).read_bytes()

        return await asyncio.to_thread(_read)

    async def delete(self, key: str) -> None:
        def _remove() -> None:
            p = self._path(key)
            if p.exists():
                p.unlink()

        await asyncio.to_thread(_remove)

    async def generate_signed_url(
        self,
        key: str,
        *,
        surface: str | None = None,
        expires_in: int = 900,
        method: str = "PUT",
    ) -> str:
        # The API layer decorates this with a base URL so the mobile client
        # can PUT to it. For local dev we route through our own endpoint.
        return f"local://{key}"


class S3Storage:
    """boto3-backed S3 storage with presigned-URL generation.

    Lazy-imports boto3 so test environments without AWS deps still work.
    """

    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        client=None,  # type: ignore[no-untyped-def]
    ) -> None:
        self.bucket = bucket
        self.region = region
        self._client = client

    @property
    def client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    async def put(self, key: str, data: bytes) -> None:
        def _do() -> None:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

        await asyncio.to_thread(_do)

    async def get(self, key: str) -> bytes:
        def _do() -> bytes:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()

        return await asyncio.to_thread(_do)

    async def delete(self, key: str) -> None:
        def _do() -> None:
            self.client.delete_object(Bucket=self.bucket, Key=key)

        await asyncio.to_thread(_do)

    async def generate_signed_url(
        self,
        key: str,
        *,
        surface: str | None = None,
        expires_in: int = 900,
        method: str = "PUT",
    ) -> str:
        client_method = "put_object" if method.upper() == "PUT" else "get_object"

        def _do() -> str:
            return self.client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
                HttpMethod=method.upper(),
            )

        return await asyncio.to_thread(_do)


_default_storage: StorageBackend | None = None


def get_default_storage() -> StorageBackend:
    """Return the process-wide storage backend, creating it on first use.

    The choice is driven by ``settings.storage_backend``:
    ``"local"`` → ``LocalFsStorage``; ``"s3"`` → ``S3Storage``.
    """
    global _default_storage
    if _default_storage is not None:
        return _default_storage

    from app.config import settings

    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError(
                "STORAGE_BACKEND=s3 requires S3_BUCKET to be configured."
            )
        _default_storage = S3Storage(bucket=settings.s3_bucket, region=settings.s3_region)
    else:
        _default_storage = LocalFsStorage(settings.storage_local_path)
    return _default_storage


def set_default_storage(backend: StorageBackend | None) -> None:
    """Override the singleton for tests. Pass ``None`` to reset."""
    global _default_storage
    _default_storage = backend
