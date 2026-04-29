import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.api import scans, verify
from app.auth import _TEST_USER
from app.db import dispose_engine, get_session_factory
from app.models import Company, User

logger = logging.getLogger(__name__)


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
    """Fire one tiny call to warm Anthropic's prompt cache for the
    verify-path system prompt.

    Anthropic's ephemeral prompt cache has a 5-minute TTL. Railway scales
    the service to zero on idle, so the first /v1/verify after each cold
    boot otherwise pays the system-prompt cache-write cost on the user's
    clock (~300–500 ms on Sonnet 4.6 with the ~2 k-token static prompt).
    A startup primer pays it on machine boot instead, where no user is
    waiting. The primer call is intentionally minimal (max_tokens=1) so
    it costs only the cache write, not a real generation.

    Failure (no API key, transient network error, rate limit) is
    swallowed — pre-warming is a latency optimisation, not a correctness
    path. The first user request will simply pay the miss it would have
    paid anyway.
    """
    try:
        from app.config import settings

        if not settings.anthropic_api_key:
            return
        from app.services.anthropic_client import build_client
        from app.services.vision import SYSTEM_PROMPT

        client = build_client(timeout=10.0)

        def _call() -> None:
            client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1,
                temperature=0.0,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": "ok"}],
            )

        await asyncio.to_thread(_call)
        logger.info("Anthropic prompt cache pre-warmed for %s", settings.anthropic_model)
    except Exception as exc:
        logger.debug("Prompt-cache pre-warm skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
