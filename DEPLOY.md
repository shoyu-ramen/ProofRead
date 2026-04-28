# Deploying Proofread

Goal: a public URL the take-home reviewer can hit. Default path is **Fly.io**
because it handles Python+uvicorn cleanly, has a generous free tier, and the
config is one file. Render and Railway are noted at the bottom as fallbacks.

## TL;DR

```bash
brew install flyctl                                    # one-time
flyctl auth login                                       # one-time
cd /Users/ross/ProofRead/backend

# Pick a globally-unique app name; edit fly.toml's `app = "..."` to match.
# (proofread-rk is the placeholder.)
flyctl apps create proofread-rk

flyctl secrets set ANTHROPIC_API_KEY=sk-ant-...
flyctl deploy
```

When deploy finishes you'll get a URL like `https://proofread-rk.fly.dev/`.
Smoke-test it:

```bash
curl -fs https://proofread-rk.fly.dev/healthz   # → {"status":"ok"}
open https://proofread-rk.fly.dev/              # → demo UI
```

The four sample-load buttons in the UI work without any further setup.

## What's already in the repo

- `backend/Dockerfile` — python:3.12-slim, copies `app/`, `pip install -e .`,
  runs uvicorn on `:8080`. Build context: `backend/` (used by Fly).
- `backend/fly.toml` — Fly v2 config: shared 1×CPU / 512 MB, auto-stop,
  HTTP health-check on `/healthz`.
- `backend/.dockerignore` — keeps `.venv/`, `tests/`, `.git/`, etc. out of
  the image so cold-builds stay under ~120 MB.
- `Dockerfile` (repo root) — same image, but `COPY`s `backend/pyproject.toml`
  + `backend/app` and uses `${PORT}`. Build context: repo root (used by
  Railway, which won't accept Dockerfiles in subdirectories).
- `railway.toml`, `.dockerignore` (repo root) — Railway-specific build /
  deploy config and ignore rules.

## Gotchas to know about

**`DATABASE_URL` is intentionally absent.** The scaffold's `app/db.py` and
`/v1/scans` flow assume Postgres, but `app/main.py`'s lifespan swallows DB
connection errors so the app boots fine without one. Result on Fly:

| Endpoint | Works without DB? |
|---|---|
| `GET /` (demo UI) | ✓ |
| `GET /healthz` | ✓ |
| `POST /v1/verify` | ✓ — this is the take-home flow |
| `POST /v1/scans` (and the rest of `/v1/scans/*`) | ✗ — 500s on DB call |

That's fine for the take-home demo. If you need `/v1/scans` later, attach a
Fly Postgres cluster: `flyctl postgres create` → `flyctl postgres attach`.

**`ANTHROPIC_API_KEY` must be set as a Fly secret, not in the image.** The
Dockerfile sets `VISION_EXTRACTOR=claude` and `ANTHROPIC_MODEL` as plain env
vars, but the API key never appears in the image. `flyctl secrets set
ANTHROPIC_API_KEY=...` injects it at runtime; without it, `/v1/verify` will
return a clean 500 with a clear message rather than crashing.

**App name is globally unique.** `proofread` and `proofread-prototype` are
likely already taken. Edit `fly.toml`'s `app = "..."` line to whatever
`flyctl apps create` accepts before running `flyctl deploy`.

**512 MB RAM is enough.** Pillow + Anthropic SDK + FastAPI is ~150 MB
resident. Bump to 1024 in `fly.toml` if you see OOM kills under load.

**Cold-start latency.** With `min_machines_running = 0`, the first request
after idle takes ~3-5 s to boot the machine. For a take-home demo where
the reviewer hits it once, that's acceptable. Set `min_machines_running = 1`
if you want consistent ≤5 s p95.

## Verifying the deploy

After `flyctl deploy` succeeds:

```bash
URL=https://proofread-rk.fly.dev   # whatever flyctl reports

# Health check
curl -fs $URL/healthz

# UI loads
curl -fsI $URL/ | head -3   # expect 200

# Static assets serve
curl -fsI $URL/static/wordmark.svg | head -3   # expect 200

# Real verify (smoke — needs ANTHROPIC_API_KEY set as a Fly secret)
curl -X POST $URL/v1/verify \
  -F "image=@artwork/labels/01_pass_old_tom_distillery.png" \
  -F "beverage_type=spirits" \
  -F "container_size_ml=750" \
  -F "is_imported=false" \
  -F 'application={"producer_record":{"brand_name":"Old Tom Distillery","class_type":"Kentucky Straight Bourbon Whiskey","alcohol_content":"45","net_contents":"750 mL","name_address":"Old Tom Distilling Co., Bardstown, Kentucky","country_of_origin":"USA"}}'
```

Expected response shape: `{"overall":"pass", "rule_results":[...], "extracted":{...}, "elapsed_ms":<3000}`.

## Iterating

```bash
# Tail logs (great for debugging the first deploy)
flyctl logs

# Re-deploy after a code change
flyctl deploy

# Rotate the API key
flyctl secrets set ANTHROPIC_API_KEY=sk-ant-new...
# (Fly restarts the machine automatically.)

# Open the app dashboard
flyctl dashboard
```

## Alternative: Render

Render's `render.yaml` equivalent — drop next to `backend/` if you'd rather
not use Fly:

```yaml
services:
  - type: web
    name: proofread
    runtime: docker
    repo: <github url>
    plan: starter
    envVars:
      - key: VISION_EXTRACTOR
        value: claude
      - key: ANTHROPIC_API_KEY
        sync: false   # set in dashboard
    healthCheckPath: /healthz
```

Then connect the repo in the Render dashboard and click Deploy. Free tier
sleeps after 15 min idle (cold start ~30 s — slower than Fly).

## Alternative: Railway

Railway builds from a GitHub-linked repo using the **repo-root** `Dockerfile`
and `railway.toml` (not `backend/Dockerfile` — Railway's Railpack rejects
Dockerfiles nested in subdirs without a Root Directory override, which is
why the two top-level files exist).

```bash
# One-time: link the repo in the Railway dashboard, then:
railway variables --set ANTHROPIC_API_KEY=sk-ant-...
git push   # Railway auto-deploys on push
```

`PORT` is injected by Railway and the root `Dockerfile`'s `CMD` expands it.
Healthcheck is `/healthz`, configured in `railway.toml`.

## If something goes wrong

- **Build fails on `pip install`:** Likely a Python version mismatch.
  `pyproject.toml` requires `>=3.12`; `Dockerfile` uses `python:3.12-slim`.
  If you're on an arm64 Mac and the prebuilt slim image is unavailable for
  some reason, switch to `python:3.12` (full image, larger but always built).
- **App boots but `/v1/verify` returns 500 with an `anthropic` error:** The
  secret isn't set or has a typo. `flyctl secrets list` shows what's
  configured (values are masked). Re-run `flyctl secrets set` and Fly will
  redeploy automatically.
- **App boots but `/` returns the API placeholder, not the UI:** The image
  isn't including `app/static/`. Check `.dockerignore` — `static/` shouldn't
  be in there (and it isn't, in the version checked in).
- **Health check failing:** `flyctl logs` will show the boot output. The
  most common cause is missing system libs for Pillow on a non-slim base.

## What this isn't

This deploy is intentionally **prototype-grade**: no Postgres, no Auth0, no
S3, no rate limiting. That matches the take-home brief ("for a prototype?
Just don't do anything crazy" — Marcus, IT). Production hardening lives in
`SPEC.md` v2/v3.
