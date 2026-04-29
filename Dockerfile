# Repo-root Dockerfile — used by Railway. Build context: repo root.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
 && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./
COPY backend/app ./app

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    VISION_EXTRACTOR=claude \
    ANTHROPIC_MODEL=claude-sonnet-4-6 \
    ANTHROPIC_HEALTH_WARNING_MODEL=claude-haiku-4-5-20251001 \
    ENABLE_HEALTH_WARNING_SECOND_PASS=true

EXPOSE 8080

# Exec form via `sh -c` so ${PORT} (set by Railway) expands at runtime while
# still letting Docker forward signals to PID 1. Falls back to 8080 locally.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
