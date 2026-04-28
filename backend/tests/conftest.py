"""Shared pytest fixtures.

Provides:

* Canonical health-warning text + a compliant-label fixture used by the
  rule and pipeline tests.
* ``db_engine`` / per-test schema setup against an in-memory SQLite via
  ``aiosqlite``. Reconfigures ``app.db`` and seeds the auth-stub user
  + company so foreign keys are satisfied.
* ``temp_storage`` — a ``LocalFsStorage`` rooted in ``tmp_path`` and
  installed as the process-wide default for the test.
* ``synthetic_label_png`` — a real PNG byte string with enough resolution
  + contrast to pass the camera-sensor capture-quality pre-flight, so
  tests that mock the OCR layer don't get rejected by sensor_check.
"""

from __future__ import annotations

import asyncio
import io
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.auth import _TEST_USER
from app.config import settings
from app.db import configure_engine, dispose_engine, get_session_factory
from app.models import Base, Company, User
from app.services.storage import LocalFsStorage, set_default_storage

CANONICAL_HEALTH_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health "
    "problems."
)


COMPLIANT_LABEL_TEXT = (
    "ANYTOWN ALE\n"
    "INDIA PALE ALE\n"
    "5.5% ABV\n"
    "12 FL OZ\n"
    "Brewed and bottled by Anytown Brewing Co., Anytown, ST 00000\n"
    + CANONICAL_HEALTH_WARNING
)


@pytest.fixture
def canonical_warning() -> str:
    return CANONICAL_HEALTH_WARNING


@pytest.fixture
def compliant_label_text() -> str:
    return COMPLIANT_LABEL_TEXT


def _reset_db_sync(db_url: str) -> None:
    """Drop + recreate the schema, then seed the test user/company.

    Run synchronously by spinning a short-lived event loop so it can be
    used from a regular pytest fixture without requiring async-mode.
    """

    async def _do() -> None:
        engine = configure_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

        factory = get_session_factory()
        async with factory() as session:
            session.add(
                Company(
                    id=_TEST_USER.company_id,
                    name="ProofRead Test Co.",
                    billing_plan="starter",
                )
            )
            session.add(
                User(
                    id=_TEST_USER.id,
                    email=_TEST_USER.email,
                    role=_TEST_USER.role,
                    company_id=_TEST_USER.company_id,
                )
            )
            await session.commit()

    asyncio.run(_do())


def _dispose_db_sync() -> None:
    asyncio.run(dispose_engine())


@pytest.fixture
def db_url(tmp_path) -> str:
    """One SQLite file per test — full isolation, no cross-test bleed."""
    return f"sqlite+aiosqlite:///{tmp_path}/test.db"


@pytest.fixture
def db_setup(db_url, monkeypatch):
    """Reset DB schema + reconfigure the global engine for this test."""
    monkeypatch.setattr(settings, "database_url", db_url)
    _reset_db_sync(db_url)
    yield db_url
    _dispose_db_sync()


@pytest.fixture
def temp_storage(tmp_path):
    """Local filesystem storage rooted at the per-test tmp dir."""
    storage = LocalFsStorage(tmp_path / "storage")
    set_default_storage(storage)
    yield storage
    set_default_storage(None)


def _make_synthetic_png(
    *,
    width: int = 1800,
    height: int = 1200,
    text: str = "TEST LABEL",
    blur: bool = False,
    glare: bool = False,
    dark: bool = False,
    bright: bool = False,
    flat: bool = False,
) -> bytes:
    """Render a real PNG that passes (or, by request, fails) sensor_check.

    Default parameters yield a sharp, well-exposed, contrasty 2.16 MP image
    that the capture-quality module judges "good". Toggles let tests force
    specific failure modes for sensor_check coverage.
    """
    img = Image.new("RGB", (width, height), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Pseudo-random sharp speckle gives the Laplacian filter something to
    # latch onto. Without this, a flat-color frame measures ~0 sharpness
    # even when "good".
    rng = random.Random(0xC0FFEE)
    for _ in range(3000):
        x = rng.randrange(width)
        y = rng.randrange(height)
        v = rng.randrange(40, 215)
        draw.rectangle((x, y, x + 4, y + 4), fill=(v, v, v))

    # Black bars to anchor contrast (simulates label print).
    for y in range(80, height - 80, 140):
        draw.rectangle((80, y, width - 80, y + 40), fill=(15, 15, 15))
    draw.text((100, 30), text, fill=(0, 0, 0))

    if blur:
        from PIL import ImageFilter

        # Radius 28 takes a sharp 2 MP frame down to ~50 Laplacian variance,
        # well under the SHARPNESS_DEGRADED threshold (120) — simulates the
        # tipsy-handheld-walking capture from SPEC §0.5.
        img = img.filter(ImageFilter.GaussianBlur(radius=28))
    if glare:
        # Wash out 60% of the frame near-white.
        overlay = Image.new("RGB", (width, height), (255, 255, 255))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle(
            (0, 0, int(width * 0.8), int(height * 0.8)), fill=200
        )
        img = Image.composite(overlay, img, mask)
    if dark:
        img = img.point(lambda p: max(0, p // 6))
    if bright:
        # Simulate photographic overexposure: high mean luminance with
        # photo-style border noise. Pushing every pixel to 255 would fold
        # three failure modes into one (overexposure, glare, AND artwork-
        # like uniform borders) and the assertions can't distinguish
        # them. Per-pixel noise centered around 235 puts the frame mean
        # comfortably above BRIGHT_DEGRADED (220), keeps the saturated-
        # pixel fraction below GLARE_DEGRADED (0.20), and leaves the
        # border with >5 stddev so the artwork heuristic doesn't catch.
        rng_b = np.random.default_rng(0xB16BD15)
        noise = rng_b.integers(225, 247, size=(height, width), dtype=np.uint8)
        img = Image.fromarray(np.stack([noise, noise, noise], axis=-1), mode="RGB")
        bd = ImageDraw.Draw(img)
        # Bright "label" content: high enough that the label region's mean
        # also lands above BRIGHT_DEGRADED, with enough internal contrast
        # for the gradient-density label-region detector to lock on.
        for y in range(80, height - 80, 140):
            bd.rectangle((80, y, width - 80, y + 40), fill=(225, 225, 225))
    if flat:
        img = Image.new("RGB", (width, height), color=(128, 128, 128))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_label_png():
    """Factory fixture — returns a callable that builds PNG bytes."""
    return _make_synthetic_png
