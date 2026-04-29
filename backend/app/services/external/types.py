"""Shared value types for the external-source tier.

`ExternalMatch` is the lowest-common-denominator shape every adapter
returns. It is deliberately narrower than the per-source response so
the L3 cache and the verify orchestrator can reason about hits without
knowing which adapter produced them; per-source detail (TTB serial
number, Open Food Facts revision, etc.) lives in `source_id` plus the
human-clickable `source_url`.

The class is frozen so a stray caller mutation cannot contaminate a
cached entry, and the to_dict / from_dict pair handles JSON
round-tripping for the persisted-cache layer (which stores serialized
records via SQLAlchemy `JSON` columns).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ExternalMatch:
    """A single match returned by an external label-data source.

    Fields are nullable where the source may legitimately omit them
    (e.g. a TTB record with no fanciful name, or an entry that has not
    yet been approved). `confidence` is the *adapter's* self-reported
    score on a 0.0–1.0 scale; the verify orchestrator can choose to
    weight or discard matches below an external floor.
    """

    source: str  # "ttb_cola" | (future: "open_food_facts", etc.)
    source_id: str  # COLA ID, OFF barcode, etc.
    brand: str | None
    fanciful_name: str | None
    class_type: str | None  # the beverage class TTB uses (e.g., "MALT BEVERAGE")
    approval_date: date | None
    label_image_url: str | None
    confidence: float  # 0.0–1.0, adapter's self-reported match score
    source_url: str | None  # human-clickable detail page

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict.

        `date` is serialized as ISO-8601 (`YYYY-MM-DD`) so the same
        round-trip works through any JSON column without a custom
        encoder. Other fields are already JSON-native.
        """
        return {
            "source": self.source,
            "source_id": self.source_id,
            "brand": self.brand,
            "fanciful_name": self.fanciful_name,
            "class_type": self.class_type,
            "approval_date": (
                self.approval_date.isoformat()
                if self.approval_date is not None
                else None
            ),
            "label_image_url": self.label_image_url,
            "confidence": self.confidence,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ExternalMatch:
        """Inverse of `to_dict`. Tolerates missing optional keys.

        The strictness here is deliberate: required identity keys
        (`source`, `source_id`, `confidence`) raise KeyError if absent
        because a record without those is structurally broken; nullable
        fields default to None so the cache layer can survive minor
        schema additions without a migration.
        """
        approval_raw = d.get("approval_date")
        approval = (
            date.fromisoformat(approval_raw)
            if isinstance(approval_raw, str)
            else None
        )
        return cls(
            source=str(d["source"]),
            source_id=str(d["source_id"]),
            brand=_opt_str(d.get("brand")),
            fanciful_name=_opt_str(d.get("fanciful_name")),
            class_type=_opt_str(d.get("class_type")),
            approval_date=approval,
            label_image_url=_opt_str(d.get("label_image_url")),
            confidence=float(d["confidence"]),  # type: ignore[arg-type]
            source_url=_opt_str(d.get("source_url")),
        )


def _opt_str(v: object) -> str | None:
    """Coerce a JSON value into `str | None`.

    JSON loads can return `None`, `str`, or a stray non-string (e.g.
    int/bool from a downstream encoder bug). Be defensive: a non-string
    non-None coerces via `str()` rather than raising, so a partially
    corrupted cache row still round-trips.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)
