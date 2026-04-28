from functools import lru_cache
from pathlib import Path

from app.config import settings


@lru_cache(maxsize=32)
def load_canonical(name: str) -> str:
    path = Path(settings.canonical_texts_path) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Canonical text not found: {path}")
    return path.read_text(encoding="utf-8").strip()
