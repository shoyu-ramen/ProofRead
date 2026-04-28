"""Synthetic 50-label corpus generator.

Composition (per harness spec):
    20 fully-compliant
    10 Health Warning typos (single-char alphabetic substitution)
    5  missing-field (one of brand_name / class_type / net_contents / name_address)
    5  missing Health Warning entirely
    5  imported with country declared (pass)
    5  imported missing country (fail on coo rule)

The corpus is deterministic given a seed: stable IDs, stable typo
positions, stable RNG choices. That stability matters because the
`measure.py` module asserts target precision/recall over the whole
corpus; flaky corpora would yield flaky thresholds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from validation.synthesize import (
    CANONICAL_HEALTH_WARNING,
    SAFE_BEER_CLASSES,
    LabelSpec,
    synthesize_label,
)

# ----------------------------------------------------------------------------
# Brand-name vocabulary
# ----------------------------------------------------------------------------

# Plausible, made-up brand fragments. Composing two chunks keeps the
# space large enough that 50 labels rarely collide on brand alone.
_BRAND_PREFIXES = [
    "Anchor", "Ironwood", "Granite", "Three Pines", "Foxtail", "Northbeam",
    "Stillwater", "Halfmoon", "Riverbend", "Black Pine", "Old Hickory",
    "Lakeshore", "Westwind", "Copper Hill", "Briarpatch", "Hightide",
    "Quietwater", "Aspen Grove", "Crooked Bear", "Brass Anchor", "Wildflower",
    "Salt Marsh", "Driftwood", "Stoneridge", "Goldfern", "Tin Roof",
]

_BRAND_SUFFIXES = [
    "Brewing Co.", "Beer Works", "Brewery", "Ales", "Brew House",
    "Craft Beer", "Cellars", "Tap Room", "Brewing", "Beer Co.",
]


def _brand(rng: random.Random) -> str:
    return f"{rng.choice(_BRAND_PREFIXES)} {rng.choice(_BRAND_SUFFIXES)}"


# ----------------------------------------------------------------------------
# Address vocabulary
# ----------------------------------------------------------------------------

_CITIES = [
    "Portland, OR", "Asheville, NC", "Burlington, VT", "Bend, OR",
    "Boulder, CO", "Athens, GA", "Madison, WI", "Bellingham, WA",
    "Petaluma, CA", "Greenville, SC", "Saratoga, NY", "Missoula, MT",
]


def _name_address(brand: str, city: str) -> str:
    return f"Brewed and bottled by {brand}, {city} 00000"


_COUNTRIES = ["Germany", "Belgium", "Mexico", "Japan", "Ireland", "Czech Republic"]


# ----------------------------------------------------------------------------
# Typo logic
# ----------------------------------------------------------------------------


def _alphabetic_substitution(text: str, rng: random.Random) -> tuple[str, int, str, str]:
    """Substitute one alphabetic character somewhere in `text`.

    Returns (mutated_text, position, original_char, replacement_char).

    Constraints:
    - Only alphabetic characters are picked, so substitution can't change
      sentence boundaries (the extractor's "trim to last period" logic
      depends on punctuation positions).
    - Replacement is a different alphabetic character of the same case;
      this guarantees the visible text stays the same length and the
      Levenshtein distance is exactly 1.
    """
    eligible = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not eligible:
        raise ValueError("No alphabetic characters to mutate")
    pos = rng.choice(eligible)
    original = text[pos]
    candidates = (
        "abcdefghijklmnopqrstuvwxyz" if original.islower()
        else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    replacement = rng.choice(
        [c for c in candidates if c != original.lower() and c != original.upper()]
    )
    if original.isupper():
        replacement = replacement.upper()
    else:
        replacement = replacement.lower()
    mutated = text[:pos] + replacement + text[pos + 1 :]
    return mutated, pos, original, replacement


# ----------------------------------------------------------------------------
# Spec construction helpers
# ----------------------------------------------------------------------------


def _baseline_spec(
    rng: random.Random, *, is_imported: bool = False, country: str | None = None
) -> LabelSpec:
    brand = _brand(rng)
    class_type = rng.choice(SAFE_BEER_CLASSES)
    abv_pct = round(rng.uniform(3.5, 9.5), 1)
    container = rng.choice([355, 473, 500, 650])
    if container == 473:
        net = "16 FL OZ"
    elif container == 500:
        net = "500 mL"
    elif container == 650:
        net = "22 FL OZ"
    else:
        net = "12 FL OZ"
    return LabelSpec(
        brand=brand,
        class_type=class_type,
        abv=f"{abv_pct}% ABV",
        net_contents=net,
        name_address=_name_address(brand, rng.choice(_CITIES)),
        health_warning_text=CANONICAL_HEALTH_WARNING,
        is_imported=is_imported,
        country=country,
        container_size_ml=container,
    )


# ----------------------------------------------------------------------------
# Corpus item
# ----------------------------------------------------------------------------


@dataclass
class CorpusItem:
    id: str
    category: str
    front_png: bytes
    back_png: bytes
    ground_truth: dict[str, str]
    ocr_text: dict[str, str]
    label_spec: LabelSpec

    def to_dict(self) -> dict[str, Any]:
        """Adapter form for clients that want the dict shape from the harness brief."""
        return {
            "id": self.id,
            "category": self.category,
            "front_png_bytes": self.front_png,
            "back_png_bytes": self.back_png,
            "ground_truth": self.ground_truth,
            "label_spec": self.label_spec,
        }


# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------

# Category sizes wired to the spec'd 50-label split. Edit here to grow the
# corpus, not at call sites.
CATEGORY_COUNTS: dict[str, int] = {
    "compliant": 20,
    "hw_typo": 10,
    "missing_field": 5,
    "hw_missing": 5,
    "imported_with_country": 5,
    "imported_no_country": 5,
}

MISSING_FIELDS = ["brand", "class_type", "net_contents", "name_address"]


def _build_compliant(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        spec = _baseline_spec(rng)
        items.append(_synthesize(f"compliant-{i:02d}", "compliant", spec, seed=10 + i))
    return items


def _build_hw_typo(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        spec = _baseline_spec(rng)
        # Reseed inside per-item so typo positions are reproducible.
        typo_rng = random.Random(20000 + i)
        mutated, pos, orig, repl = _alphabetic_substitution(
            CANONICAL_HEALTH_WARNING, typo_rng
        )
        spec.health_warning_text = mutated
        spec.metadata = {"typo_position": pos, "typo_original": orig, "typo_replacement": repl}
        items.append(_synthesize(f"hw_typo-{i:02d}", "hw_typo", spec, seed=200 + i))
    return items


def _build_missing_field(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        spec = _baseline_spec(rng)
        # Cycle through fields to guarantee coverage of all four with n=5.
        field = MISSING_FIELDS[i % len(MISSING_FIELDS)]
        setattr(spec, field, None)
        spec.metadata = {"missing_field": field}
        items.append(_synthesize(f"missing_field-{i:02d}", "missing_field", spec, seed=400 + i))
    return items


def _build_hw_missing(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        spec = _baseline_spec(rng)
        spec.health_warning_text = None
        items.append(_synthesize(f"hw_missing-{i:02d}", "hw_missing", spec, seed=600 + i))
    return items


def _build_imported_with_country(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        country = _COUNTRIES[i % len(_COUNTRIES)]
        spec = _baseline_spec(rng, is_imported=True, country=country)
        items.append(
            _synthesize(
                f"imported_with_country-{i:02d}",
                "imported_with_country",
                spec,
                seed=800 + i,
            )
        )
    return items


def _build_imported_no_country(rng: random.Random, n: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for i in range(n):
        spec = _baseline_spec(rng, is_imported=True, country=None)
        items.append(
            _synthesize(f"imported_no_country-{i:02d}", "imported_no_country", spec, seed=1000 + i)
        )
    return items


_BUILDERS = {
    "compliant": _build_compliant,
    "hw_typo": _build_hw_typo,
    "missing_field": _build_missing_field,
    "hw_missing": _build_hw_missing,
    "imported_with_country": _build_imported_with_country,
    "imported_no_country": _build_imported_no_country,
}


def _synthesize(item_id: str, category: str, spec: LabelSpec, seed: int) -> CorpusItem:
    front_png, back_png, ground_truth, ocr_text = synthesize_label(spec, seed=seed)
    return CorpusItem(
        id=item_id,
        category=category,
        front_png=front_png,
        back_png=back_png,
        ground_truth=ground_truth,
        ocr_text=ocr_text,
        label_spec=spec,
    )


def generate_corpus(seed: int = 1234) -> list[CorpusItem]:
    """Generate the full 50-label corpus deterministically.

    Categories are emitted in the same order as `CATEGORY_COUNTS`. Within a
    category, items are seeded by index so reordering categories does not
    change the resulting items.
    """
    rng = random.Random(seed)
    items: list[CorpusItem] = []
    for category, count in CATEGORY_COUNTS.items():
        items.extend(_BUILDERS[category](rng, count))
    return items


def corpus_summary(items: list[CorpusItem]) -> dict[str, int]:
    """Tally items per category — useful for harness output."""
    summary: dict[str, int] = {}
    for item in items:
        summary[item.category] = summary.get(item.category, 0) + 1
    summary["total"] = len(items)
    return summary
