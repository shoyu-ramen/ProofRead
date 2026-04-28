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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await ensure_test_user()
    except Exception:
        # Don't block startup if the DB isn't reachable yet — health
        # checks will surface the issue. Tests pre-seed before TestClient
        # is constructed, so this is a no-op there.
        pass
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
