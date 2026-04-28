import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.rules.types import Bbox


@dataclass
class OCRBlock:
    text: str
    bbox: Bbox
    confidence: float = 0.99


@dataclass
class OCRResult:
    full_text: str
    blocks: list[OCRBlock]
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


class OCRProvider(Protocol):
    def process(self, image_bytes: bytes, hint: str | None = None) -> OCRResult: ...


class MockOCRProvider:
    """Returns a fixed OCRResult from a fixture dict or JSON file. Used in tests."""

    def __init__(self, fixture: dict[str, Any] | str | Path):
        if isinstance(fixture, (str, Path)):
            fixture = json.loads(Path(fixture).read_text(encoding="utf-8"))
        assert isinstance(fixture, dict)
        self._fixture: dict[str, Any] = fixture

    def process(self, image_bytes: bytes, hint: str | None = None) -> OCRResult:
        blocks = [
            OCRBlock(
                text=b["text"],
                bbox=tuple(b["bbox"]),  # type: ignore[arg-type]
                confidence=b.get("confidence", 0.99),
            )
            for b in self._fixture.get("blocks", [])
        ]
        return OCRResult(
            full_text=self._fixture["full_text"],
            blocks=blocks,
            provider="mock",
            raw=self._fixture,
        )


class GoogleVisionOCRProvider:
    """Production OCR via Google Cloud Vision document_text_detection.

    Requires the `[google-vision]` extra and Application Default Credentials
    (e.g. GOOGLE_APPLICATION_CREDENTIALS env var pointing at a JSON keyfile).
    """

    def __init__(self) -> None:
        from google.cloud import vision  # noqa: F401  (lazy)
        from google.cloud import vision as _vision

        self._client = _vision.ImageAnnotatorClient()

    def process(self, image_bytes: bytes, hint: str | None = None) -> OCRResult:
        from google.cloud import vision

        image = vision.Image(content=image_bytes)
        response = self._client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")

        annotation = response.full_text_annotation
        full_text = annotation.text or ""
        blocks: list[OCRBlock] = []
        for page in annotation.pages:
            for block in page.blocks:
                vertices = block.bounding_box.vertices
                xs = [v.x for v in vertices] or [0]
                ys = [v.y for v in vertices] or [0]
                bbox: Bbox = (
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                )
                text = " ".join(
                    "".join(s.text for s in word.symbols)
                    for paragraph in block.paragraphs
                    for word in paragraph.words
                )
                blocks.append(OCRBlock(text=text, bbox=bbox, confidence=block.confidence))
        return OCRResult(full_text=full_text, blocks=blocks, provider="google_vision", raw={})


def get_default_provider() -> OCRProvider:
    from app.config import settings

    if settings.ocr_provider == "google_vision":
        return GoogleVisionOCRProvider()
    raise RuntimeError(
        "OCR_PROVIDER=mock requires an explicit MockOCRProvider; "
        "wire one up in tests or set OCR_PROVIDER=google_vision in production."
    )
