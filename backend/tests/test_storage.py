"""Tests for the storage abstraction.

LocalFsStorage is exercised end-to-end. S3Storage's presigned-URL
generation is tested with a stub boto3 client so we don't need real
AWS credentials.
"""

from __future__ import annotations

import pytest

from app.services.storage import (
    LocalFsStorage,
    S3Storage,
    get_default_storage,
    scan_image_key,
    set_default_storage,
)


@pytest.mark.asyncio
async def test_local_storage_round_trip(tmp_path):
    storage = LocalFsStorage(tmp_path / "blobs")
    await storage.put("scans/abc/front", b"hello")
    assert await storage.get("scans/abc/front") == b"hello"


@pytest.mark.asyncio
async def test_local_storage_signed_url_routes_through_app(tmp_path):
    storage = LocalFsStorage(tmp_path)
    url = await storage.generate_signed_url("scans/abc/front", surface="front")
    assert url.startswith("local://")


@pytest.mark.asyncio
async def test_local_storage_delete(tmp_path):
    storage = LocalFsStorage(tmp_path)
    await storage.put("k", b"x")
    await storage.delete("k")
    with pytest.raises(FileNotFoundError):
        await storage.get("k")


@pytest.mark.asyncio
async def test_local_storage_rejects_path_escape(tmp_path):
    storage = LocalFsStorage(tmp_path)
    with pytest.raises(ValueError):
        await storage.put("../escape", b"x")


def test_scan_image_key_shape():
    assert scan_image_key("abc-123", "front") == "scans/abc-123/front"
    assert scan_image_key("abc-123", "back") == "scans/abc-123/back"


class _StubS3Client:
    def __init__(self):
        self.calls = []

    def generate_presigned_url(self, **kwargs):
        self.calls.append(("generate_presigned_url", kwargs))
        return f"https://example.com/{kwargs['Params']['Key']}?sig=stub"

    def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))

    def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))

        class _Body:
            def read(self):
                return b"stub-bytes"

        return {"Body": _Body()}


@pytest.mark.asyncio
async def test_s3_storage_generates_presigned_put_url():
    stub = _StubS3Client()
    storage = S3Storage(bucket="my-bucket", region="us-east-1", client=stub)
    url = await storage.generate_signed_url(
        "scans/x/front", surface="front", method="PUT"
    )
    assert "my-bucket" not in url  # the stub puts the key in the URL, not the bucket
    assert "scans/x/front" in url
    method, params = stub.calls[0]
    assert method == "generate_presigned_url"
    assert params["ClientMethod"] == "put_object"
    assert params["Params"]["Bucket"] == "my-bucket"
    assert params["Params"]["Key"] == "scans/x/front"


@pytest.mark.asyncio
async def test_s3_storage_put_get_round_trip():
    stub = _StubS3Client()
    storage = S3Storage(bucket="b", region="us-east-1", client=stub)
    await storage.put("k", b"payload")
    got = await storage.get("k")
    assert got == b"stub-bytes"  # the stub returns a fixed body
    assert stub.calls[0][0] == "put_object"
    assert stub.calls[0][1]["Body"] == b"payload"


def test_default_storage_factory_local(tmp_path, monkeypatch):
    from app.config import settings

    set_default_storage(None)
    monkeypatch.setattr(settings, "storage_backend", "local")
    monkeypatch.setattr(settings, "storage_local_path", str(tmp_path))
    backend = get_default_storage()
    assert isinstance(backend, LocalFsStorage)
    set_default_storage(None)


def test_default_storage_factory_s3_requires_bucket(monkeypatch):
    from app.config import settings

    set_default_storage(None)
    monkeypatch.setattr(settings, "storage_backend", "s3")
    monkeypatch.setattr(settings, "s3_bucket", None)
    with pytest.raises(RuntimeError, match="S3_BUCKET"):
        get_default_storage()
    set_default_storage(None)
