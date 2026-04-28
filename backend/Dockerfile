FROM python:3.12-slim

WORKDIR /app

# Pillow needs libjpeg + zlib at runtime; nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
 && rm -rf /var/lib/apt/lists/*

# Project metadata + source. pyproject's package finder (include = ["app*"])
# requires the app/ tree to be present at install time, so copy both before
# pip install rather than splitting into separate layers.
COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    VISION_EXTRACTOR=claude \
    ANTHROPIC_MODEL=claude-opus-4-7

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
