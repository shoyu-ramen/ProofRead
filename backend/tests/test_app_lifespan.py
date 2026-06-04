"""Coverage for app startup/shutdown wiring (app/main.py) + the lazy
engine accessors (app/db.py).

`app/main.py` sat at ~36% because every other test constructs
`TestClient(app)` *without* the `with` context manager, so Starlette
never enters the `lifespan` — none of the startup (rule-load, telemetry
init, migration, user-seed, prompt-cache primer) or shutdown (pool +
engine teardown) code ran. Entering the context once exercises the whole
path; the migration step intentionally fails-and-swallows against the
async SQLite test URL, which is itself a branch worth covering.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.db import (
    configure_engine,
    dispose_engine,
    get_engine,
    get_session_factory,
)
from app.main import _prewarm_prompt_cache, app, ensure_test_user
from app.models import Base, User
from app.auth import _TEST_USER


def test_lifespan_startup_and_shutdown(db_setup, temp_storage, monkeypatch):
    # Empty API key → the prompt-cache primer no-ops instead of firing a
    # real Anthropic round-trip from the background task.
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    # Stub the migration step to raise so we cover the lifespan's
    # fail-open try/except (the realistic Railway-without-Postgres path)
    # WITHOUT running the real `alembic upgrade`. Real alembic calls
    # `logging.config.fileConfig()` from alembic.ini with
    # `disable_existing_loggers=True`, which silences every already-
    # configured app logger for the rest of the pytest process and breaks
    # downstream `caplog`-based tests. Exercising the except branch here
    # gives the same coverage without that global side effect.
    def _boom() -> None:
        raise RuntimeError("simulated: DATABASE_URL unreachable")

    monkeypatch.setattr("app.main._apply_alembic_migrations_sync", _boom)

    with TestClient(app) as client:
        # Startup ran: rules validated, telemetry init'd (no-op locally),
        # migration attempted (swallowed on the async SQLite URL), test
        # user ensured, primer task scheduled.
        assert client.get("/healthz").json() == {"status": "ok"}
        assert app.state.prompt_cache_primer is not None
    # Exiting the context ran shutdown_pool() + dispose_engine() cleanly.


def test_index_serves_static_or_placeholder(db_setup, temp_storage):
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Proofread API" in res.text or "<!" in res.text or "<html" in res.text.lower()


@pytest.mark.asyncio
async def test_ensure_test_user_inserts_then_is_idempotent(monkeypatch, tmp_path):
    """First call inserts the stub company + user; second is a no-op.

    Runs against a fresh, unseeded schema so both the insert branch and
    the already-exists branch are covered (conftest's `db_setup` pre-seeds,
    so it only ever hits the latter).
    """
    url = f"sqlite+aiosqlite:///{tmp_path}/fresh.db"
    monkeypatch.setattr(settings, "database_url", url)
    engine = configure_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await ensure_test_user()  # insert branch
    factory = get_session_factory()
    async with factory() as session:
        assert (
            await session.scalar(select(User).where(User.id == _TEST_USER.id))
        ) is not None

    await ensure_test_user()  # exists branch — must not raise or duplicate
    async with factory() as session:
        users = (await session.scalars(select(User))).all()
        assert len(users) == 1

    await dispose_engine()


@pytest.mark.asyncio
async def test_prewarm_noops_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    # Must return immediately without importing/constructing a client.
    await _prewarm_prompt_cache()


@pytest.mark.asyncio
async def test_prewarm_primes_both_prompts_with_cache_breakpoint(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    calls: list[dict] = []

    class _FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return None

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(
        "app.services.anthropic_client.build_client",
        lambda timeout=10.0: _FakeClient(),
    )

    await _prewarm_prompt_cache()

    # Both the primary extractor prompt and the second-pass prompt are
    # primed, each with an ephemeral cache breakpoint.
    assert len(calls) == 2
    for kw in calls:
        assert kw["max_tokens"] == 1
        assert kw["system"][0]["cache_control"]["type"] == "ephemeral"


@pytest.mark.asyncio
async def test_get_engine_and_factory_lazily_configure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{tmp_path}/lazy.db"
    )
    # Force both globals to None so the lazy-configure branches fire.
    await dispose_engine()
    assert get_session_factory() is not None  # covers _SessionLocal-None branch

    await dispose_engine()
    assert get_engine() is not None  # covers _engine-None branch

    await dispose_engine()
