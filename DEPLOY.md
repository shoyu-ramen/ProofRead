# Deploying Proofread

Railway only. Single Dockerfile at the repo root, auto-deploys on `git push`.

Live: <https://proofread-rk-production.up.railway.app>

## TL;DR

```bash
# One-time (Railway dashboard):
#   1. New Project → Deploy from GitHub repo → pick this repo
#   2. Service Settings → leave Root Directory blank
#   3. Variables → set ANTHROPIC_API_KEY=sk-ant-...
#   4. Settings → Networking → Generate Domain

# After that, every push auto-deploys:
git push
```

```bash
URL=https://proofread-rk-production.up.railway.app

curl -fs $URL/healthz   # → {"status":"ok"}
open $URL/              # → demo UI
```

## What's in the repo

- `Dockerfile` (repo root) — `python:3.12-slim`, copies
  `backend/pyproject.toml` + `backend/app`, installs, runs uvicorn with
  `${PORT:-8080}`. Build context: repo root.
- `.dockerignore` (repo root) — strips `mobile/`, `artwork/`, screenshots,
  tests, caches.
- `railway.toml` — forces the Dockerfile builder, healthcheck `/healthz`,
  restart `ON_FAILURE`.

## Required variables

| Name | Where | What |
|---|---|---|
| `ANTHROPIC_API_KEY` | Railway Variables | API key for the Claude vision extractor. |
| `PORT` | injected by Railway | Bound port; Dockerfile expands `${PORT:-8080}`. |

The image bakes in:

- `VISION_EXTRACTOR=claude`
- `ANTHROPIC_MODEL=claude-sonnet-4-6`
- `ANTHROPIC_HEALTH_WARNING_MODEL=claude-haiku-4-5-20251001`
- `ENABLE_HEALTH_WARNING_SECOND_PASS=true`

Override via Railway Variables.

## Gotchas

**No `DATABASE_URL`.** `app/main.py`'s lifespan swallows DB connection
errors. Result on Railway: `/`, `/healthz`, `/v1/verify` all work; `/v1/scans`
500s. Fine for the demo. Add a Postgres plugin + set `DATABASE_URL` for the
scan flow.

**`ANTHROPIC_API_KEY` lives in Railway Variables, not the image.** Without
it, `/v1/verify` returns a clean 503.

**Cold starts.** Railway scales to zero by default — first request after
idle takes ~3–5 s. Bump min-instances in Service Settings for tighter p95.

## Verifying the deploy

```bash
URL=https://proofread-rk-production.up.railway.app

curl -fs $URL/healthz
curl -fsI $URL/ | head -3
curl -fsI $URL/static/wordmark.svg | head -3

curl -X POST $URL/v1/verify \
  -F "image=@artwork/labels/01_pass_old_tom_distillery.png" \
  -F "beverage_type=spirits" \
  -F "container_size_ml=750" \
  -F "is_imported=false" \
  -F 'application={"producer_record":{"brand_name":"Old Tom Distillery","class_type":"Kentucky Straight Bourbon Whiskey","alcohol_content":"45","net_contents":"750 mL","name_address":"Old Tom Distilling Co., Bardstown, Kentucky","country_of_origin":"USA"}}'
```

Expected: `{"overall":"pass", "rule_results":[...], "extracted":{...}, "elapsed_ms":<3000}`.

## Iterating

```bash
railway logs                 # tail
git push                      # redeploy
railway variables --set ANTHROPIC_API_KEY=sk-ant-new...
railway open
```

## If it breaks

- **Build fails on `pip install`:** Python version mismatch.
- **`/v1/verify` returns 503 `vision_unavailable`:** `ANTHROPIC_API_KEY`
  unset or typo'd.
- **`/` returns the placeholder:** `app/static/` excluded by mistake; check
  `.dockerignore`.
- **Health check fails:** Check Railway logs; usually missing system libs
  for Pillow on a non-slim base, or `${PORT}` not expanding under a
  non-shell `CMD`.
- **Wrong language detected:** Service Settings → Builder must be
  "Dockerfile", Root Directory empty.

## Mobile

`mobile/eas.json` sets `EXPO_PUBLIC_API_BASE_URL` to the Railway URL for
preview/production builds. The mobile client falls back to
`extra.apiBaseUrl` in `app.json` (LAN dev IP) when the env var isn't
set — that's the dev-mode path.

## What this isn't

Prototype-grade. No Postgres, Auth0, S3, or rate limiting. Production
hardening lives in `SPEC.md` v2/v3.
