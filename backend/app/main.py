import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.api import admin, scans, verify
from app.auth import _TEST_USER
from app.db import dispose_engine, get_session_factory
from app.models import Company, User

logger = logging.getLogger(__name__)


def _apply_alembic_migrations_sync() -> None:
    """Run ``alembic upgrade head`` against the configured DATABASE_URL.

    Runs synchronously in a worker thread (called via
    ``asyncio.to_thread`` from the lifespan) because Alembic's
    ``command.upgrade`` is sync-only and spawns its own short-lived
    engine. We bind the config to ``settings.database_url`` so the
    migration uses the same URL the application will use post-startup.

    Failures are caught by the caller's try/except — the lifespan must
    not block ``/healthz`` from answering 200 just because the DB is
    unreachable on a Railway demo deploy without a Postgres plugin.
    The L3 cache and persistent enrichment paths each gate themselves
    on the DB being reachable, so a missing migration only affects
    those tiers.
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    from app.config import settings

    # Locate alembic.ini relative to the backend package — works under
    # both the local checkout (`backend/alembic.ini`) and the Railway
    # image where /app is the backend root and alembic.ini sits in /app.
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "alembic.ini"
    if not config_path.exists():
        # Fallback for callers running from a different layout (e.g.
        # `pytest` invoked from the worktree root) — let Alembic's
        # default loader resolve the file from the cwd.
        config = Config()
    else:
        config = Config(str(config_path))
    # Force the alembic env.py to read this URL rather than re-importing
    # `app.config.settings` — a test-time monkeypatch of the URL stays
    # honored that way.
    config.set_main_option("sqlalchemy.url", settings.database_url)
    # `script_location` resolves relative to the ini's directory; if we
    # had to fall back to a default Config above, point it at the right
    # alembic dir explicitly so the migrations are still found.
    if not config_path.exists():
        config.set_main_option(
            "script_location", str(repo_root / "alembic")
        )
    command.upgrade(config, "head")


async def ensure_test_user() -> None:
    """Seed the auth-stub user + company so FK constraints are satisfied.

    Idempotent — safe to call on every startup. The real auth flow will
    upsert the user during the Auth0 token-exchange handshake.
    """
    factory = get_session_factory()
    async with factory() as session:
        existing_company = await session.scalar(
            select(Company).where(Company.id == _TEST_USER.company_id)
        )
        if existing_company is None:
            session.add(
                Company(
                    id=_TEST_USER.company_id,
                    name="ProofRead Test Co.",
                    billing_plan="starter",
                )
            )
        existing_user = await session.scalar(
            select(User).where(User.id == _TEST_USER.id)
        )
        if existing_user is None:
            session.add(
                User(
                    id=_TEST_USER.id,
                    email=_TEST_USER.email,
                    role=_TEST_USER.role,
                    company_id=_TEST_USER.company_id,
                )
            )
        await session.commit()


async def _prewarm_prompt_cache() -> None:
    """Fire tiny calls to warm Anthropic's prompt cache for both verify-path
    system prompts (primary extractor + Government-Warning second-pass).

    Anthropic's ephemeral prompt cache has a 5-minute TTL. Railway scales
    the service to zero on idle, so the first /v1/verify after each cold
    boot otherwise pays the system-prompt cache-write cost on the user's
    clock (~300–500 ms on Sonnet 4.6 with the ~2 k-token static prompt).
    Both prompts are warmed because the second-pass runs concurrently with
    the primary on every cold call — leaving the second-pass unprimed
    would still cost the user ~200–400 ms even with the primary primed
    (the slower of the two wins). And with the recent SPEC §0.5 redundancy
    restoration the second-pass is on a different model family from the
    primary, so each has its own cache breakpoint to write.

    Each primer call is intentionally minimal (max_tokens=1) so it costs
    only the cache write. The two run concurrently so total startup
    latency is bounded by the slower one.

    Failure (no API key, transient network error, rate limit) is
    swallowed per-prompt — pre-warming is a latency optimisation, not a
    correctness path. The first user request will simply pay the miss it
    would have paid anyway for whichever prompt's primer was lost.
    """
    from app.config import settings

    if not settings.anthropic_api_key:
        return
    from app.services.anthropic_client import build_client
    from app.services.health_warning_second_pass import (
        SYSTEM_PROMPT as SECOND_PASS_PROMPT,
    )
    from app.services.vision import SYSTEM_PROMPT as PRIMARY_PROMPT

    client = build_client(timeout=10.0)

    def _prime(model: str, system_prompt: str, label: str) -> None:
        try:
            client.messages.create(
                model=model,
                max_tokens=1,
                temperature=0.0,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": "ok"}],
            )
            logger.info(
                "Anthropic prompt cache pre-warmed for %s (%s)", model, label
            )
        except Exception as exc:
            logger.debug("Prompt-cache pre-warm skipped (%s): %s", label, exc)

    await asyncio.gather(
        asyncio.to_thread(_prime, settings.anthropic_model, PRIMARY_PROMPT, "primary"),
        asyncio.to_thread(
            _prime,
            settings.anthropic_health_warning_model,
            SECOND_PASS_PROMPT,
            "second-pass",
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate the rule definitions before the API accepts traffic. The
    # loader's fail-fast checks (unknown check types, malformed
    # applies_if/exempt_if, duplicate rule_ids) would otherwise only
    # surface on the first scan to hit the affected rule.
    from app.rules.loader import load_rules

    load_rules()

    # Telemetry init runs as early as possible so any startup errors after
    # this line are captured. Both helpers no-op silently when the
    # corresponding env vars / packages are missing — local dev sees
    # nothing extra. OTel auto-instruments the FastAPI app when init
    # succeeds, so every route becomes a span without per-handler code.
    from app.telemetry import init_otel, init_sentry

    init_sentry()
    init_otel(app)

    # Apply DB schema migrations before seeding or accepting traffic.
    # Idempotent: ``alembic upgrade head`` is a no-op when the DB is
    # already at the target revision, and tests that pre-create their
    # own SQLite schema via ``Base.metadata.create_all`` short-circuit
    # this because the alembic_version table they create matches.
    # Failures (no DATABASE_URL, transient connection error) are
    # swallowed for the same reason ``ensure_test_user`` is — Railway's
    # demo deploy keeps ``/`` and ``/healthz`` working without Postgres
    # configured. The persisted-cache + scans paths each fail open
    # downstream when the schema isn't present.
    try:
        await asyncio.to_thread(_apply_alembic_migrations_sync)
    except Exception:
        logger.warning(
            "Alembic migration step failed; persisted cache + scans "
            "tier may be unavailable until DATABASE_URL is reachable.",
            exc_info=True,
        )

    try:
        await ensure_test_user()
    except Exception:
        # Don't block startup if the DB isn't reachable yet — health
        # checks will surface the issue. Tests pre-seed before TestClient
        # is constructed, so this is a no-op there.
        pass
    # Fire the prompt-cache primer in the background so startup isn't
    # blocked by an Anthropic round-trip. The handle is held by the
    # event loop; we don't await it because (a) we don't need its result
    # and (b) a transient API failure must not delay /healthz answering
    # ready. Stored on `app.state` so a stray reference keeps the task
    # from being GC'd mid-flight while preserving fire-and-forget.
    app.state.prompt_cache_primer = asyncio.create_task(_prewarm_prompt_cache())
    yield
    # Tear down the verify-pipeline thread pool before the engine so the
    # workers don't try to use a disposed DB session in flight.
    from app.services.verify import shutdown_pool

    shutdown_pool()
    await dispose_engine()


app = FastAPI(
    title="Proofread API",
    version="0.2.0",
    description=(
        "TTB label compliance verification. /v1/scans is the multi-image "
        "OCR-based scan flow (v1 beer); /v1/verify is the single-shot "
        "Claude-vision flow used by the agent UI."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(scans.router, prefix="/v1")
app.include_router(verify.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")


_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the demo UI from /static/index.html. Returns a placeholder if
    the static asset hasn't been built (e.g. running an API-only image)."""
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(index_html.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Proofread API</h1>"
        "<p>UI assets not present at <code>app/static/index.html</code>. "
        "POST <code>/v1/verify</code> to use the API directly.</p>"
    )
