import re

from rapidfuzz import fuzz

from app.rules.types import Bbox, ExtractedField, ExtractionContext
from app.services.ocr import OCRBlock, OCRResult

HEALTH_WARNING_ANCHOR = "GOVERNMENT WARNING"

# The canonical Health Warning ends with "may cause health problems."
# Anchoring on this phrase lets us trim trailing label text (recycling
# notices, social-responsibility boilerplate) that would otherwise pollute
# the exact-text comparison and produce a false-fail.
_HW_TERMINAL_RE = re.compile(r"health\s+problems\.?", re.IGNORECASE)

# Score cutoff for the fuzzy anchor match. ~85 corresponds to ≤2-3 char
# edits in the 18-char anchor — covers single-letter OCR drops like
# "GOVERMENT WARNING" without admitting unrelated text.
_HW_ANCHOR_SCORE_CUTOFF = 85.0

# Ordered longest-first so multi-word classes match before constituent words.
BEER_CLASSES = [
    "imperial stout",
    "india pale ale",
    "session ipa",
    "double ipa",
    "new england ipa",
    "amber ale",
    "brown ale",
    "pale ale",
    "milk stout",
    "cream ale",
    "barrel-aged",
    "barleywine",
    "doppelbock",
    "quadrupel",
    "hefeweizen",
    "weissbier",
    "pilsener",
    "pilsner",
    "porter",
    "saison",
    "tripel",
    "weizen",
    "dubbel",
    "lambic",
    "kolsch",
    "wheat",
    "stout",
    "lager",
    "gose",
    "bock",
    "sour",
    "neipa",
    "ipa",
    "ale",
    "malt liquor",
    "cider",
]

ABV_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%\s*(?:ABV|ALC\.?|ALCOHOL\b)?",
    re.IGNORECASE,
)
NET_CONTENTS_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:ml|fl\.?\s*oz|fluid\s+ounces?|liters?|l)\b",
    re.IGNORECASE,
)
NAME_ADDRESS_RE = re.compile(
    r"(?:brewed\s+(?:and\s+bottled\s+)?by|bottled\s+by|imported\s+by|"
    r"produced\s+(?:and\s+bottled\s+)?by|distributed\s+by)\s+[^\n]{1,200}",
    re.IGNORECASE,
)
COUNTRY_RE = re.compile(
    r"(?:product\s+of|imported\s+from|country\s+of\s+origin\s*[:.]?)\s+([A-Za-z][A-Za-z\s]{2,30})",
    re.IGNORECASE,
)


def extract_beer_fields(
    ocr_results: dict[str, OCRResult],
    container_size_ml: int,
    is_imported: bool = False,
) -> ExtractionContext:
    raw_texts = {surface: r.full_text for surface, r in ocr_results.items()}

    fields: dict[str, ExtractedField] = {}

    hw = _extract_health_warning(ocr_results)
    if hw:
        fields["health_warning"] = hw

    abv = _find_first(ABV_RE, ocr_results)
    if abv:
        fields["alcohol_content"] = abv

    net = _find_first(NET_CONTENTS_RE, ocr_results)
    if net:
        fields["net_contents"] = net

    addr = _find_first(NAME_ADDRESS_RE, ocr_results)
    if addr:
        fields["name_address"] = addr

    if is_imported:
        country = _find_first(COUNTRY_RE, ocr_results, group=1)
        if country:
            fields["country_of_origin"] = country

    cls = _extract_class_type(ocr_results)
    if cls:
        fields["class_type"] = cls

    brand = _extract_brand_name(ocr_results)
    if brand:
        fields["brand_name"] = brand

    abv_pct = _parse_abv_pct(fields.get("alcohol_content"))

    return ExtractionContext(
        fields=fields,
        beverage_type="beer",
        container_size_ml=container_size_ml,
        is_imported=is_imported,
        abv_pct=abv_pct,
        raw_ocr_texts=raw_texts,
    )


def _find_first(
    pattern: re.Pattern[str],
    ocr_results: dict[str, OCRResult],
    group: int = 0,
) -> ExtractedField | None:
    for surface, ocr in ocr_results.items():
        m = pattern.search(ocr.full_text)
        if m:
            value = (m.group(group) if group else m.group(0)).strip()
            bbox = _bbox_for_text(ocr.blocks, m.group(0))
            return ExtractedField(
                value=value,
                bbox=bbox,
                confidence=0.9,
                source_image_id=surface,
            )
    return None


def _extract_health_warning(ocr_results: dict[str, OCRResult]) -> ExtractedField | None:
    for surface, ocr in ocr_results.items():
        text = ocr.full_text
        idx = _find_hw_anchor(text.upper())
        if idx == -1:
            continue
        snippet = _trim_hw_snippet(text[idx : idx + 400])
        bbox = _bbox_for_text(ocr.blocks, HEALTH_WARNING_ANCHOR)
        return ExtractedField(
            value=snippet,
            bbox=bbox,
            confidence=0.95,
            source_image_id=surface,
        )
    return None


def _find_hw_anchor(text_upper: str) -> int:
    """Locate the Health Warning anchor with tolerance for single-letter
    OCR drops (e.g. "GOVERMENT WARNING"). Returns -1 if not found."""
    idx = text_upper.find(HEALTH_WARNING_ANCHOR)
    if idx != -1:
        return idx
    result = fuzz.partial_ratio_alignment(
        HEALTH_WARNING_ANCHOR, text_upper, score_cutoff=_HW_ANCHOR_SCORE_CUTOFF
    )
    if result is None:
        return -1
    return result.dest_start


def _trim_hw_snippet(snippet: str) -> str:
    """Trim the captured snippet at the canonical's terminal phrase.

    Falls back to the last period in the window only when the phrase isn't
    found, which is the rare case of OCR mangling the closing words.
    """
    m = _HW_TERMINAL_RE.search(snippet)
    if m:
        return snippet[: m.end()]
    last_period = snippet.rfind(".")
    if last_period >= 200:
        return snippet[: last_period + 1]
    return snippet


def _extract_class_type(ocr_results: dict[str, OCRResult]) -> ExtractedField | None:
    for surface, ocr in ocr_results.items():
        text_lower = ocr.full_text.lower()
        for cls in BEER_CLASSES:
            if re.search(rf"\b{re.escape(cls)}\b", text_lower):
                bbox = _bbox_for_text(ocr.blocks, cls)
                return ExtractedField(
                    value=cls,
                    bbox=bbox,
                    confidence=0.7,
                    source_image_id=surface,
                )
    return None


def _extract_brand_name(ocr_results: dict[str, OCRResult]) -> ExtractedField | None:
    front = ocr_results.get("front")
    if front and front.blocks:
        biggest = max(front.blocks, key=lambda b: b.bbox[2] * b.bbox[3])
        return ExtractedField(
            value=biggest.text.strip(),
            bbox=biggest.bbox,
            confidence=0.6,
            source_image_id="front",
        )
    # Fallback: pick the largest block across any surface.
    best: tuple[str, OCRBlock] | None = None
    for surface, ocr in ocr_results.items():
        if not ocr.blocks:
            continue
        candidate = max(ocr.blocks, key=lambda b: b.bbox[2] * b.bbox[3])
        candidate_area = candidate.bbox[2] * candidate.bbox[3]
        if best is None or candidate_area > best[1].bbox[2] * best[1].bbox[3]:
            best = (surface, candidate)
    if best is None:
        return None
    surface, blk = best
    return ExtractedField(
        value=blk.text.strip(),
        bbox=blk.bbox,
        confidence=0.5,
        source_image_id=surface,
    )


def _bbox_for_text(blocks: list[OCRBlock], snippet: str) -> Bbox | None:
    snippet_lower = snippet.strip().lower()
    if not snippet_lower:
        return None
    matching = [b for b in blocks if snippet_lower in b.text.lower()]
    if matching:
        return _union_bbox(matching)

    # Word-level fallback for snippets split across multiple OCR blocks
    # (e.g. "GOVERNMENT WARNING" in two adjacent blocks). Require all
    # substantive words to be covered by a *spatially coherent* cluster
    # so that an unrelated block elsewhere on the label sharing one word
    # ("WARNING: keep cold") doesn't pollute the highlight bbox.
    words = [w for w in re.split(r"\W+", snippet_lower) if len(w) > 2]
    if not words:
        return None
    cluster = _word_cluster(blocks, words)
    if cluster is None:
        return None
    return _union_bbox(cluster)


def _union_bbox(blocks: list[OCRBlock]) -> Bbox:
    xs = [b.bbox[0] for b in blocks]
    ys = [b.bbox[1] for b in blocks]
    rights = [b.bbox[0] + b.bbox[2] for b in blocks]
    bottoms = [b.bbox[1] + b.bbox[3] for b in blocks]
    return (min(xs), min(ys), max(rights) - min(xs), max(bottoms) - min(ys))


def _word_cluster(
    blocks: list[OCRBlock], words: list[str]
) -> list[OCRBlock] | None:
    """Return a spatially-coherent block set that collectively covers all
    `words`, or None. Anchors on the first occurrence of the first word and
    only considers blocks within ~4 median line-heights vertically. Then
    picks a greedy minimal cover so the bbox doesn't span more than needed.
    """
    anchor = next((b for b in blocks if words[0] in b.text.lower()), None)
    if anchor is None:
        return None

    heights = sorted(b.bbox[3] for b in blocks if b.bbox[3] > 0)
    if not heights:
        return None
    median_h = heights[len(heights) // 2]
    max_dy = max(median_h * 4, 1)

    ay = anchor.bbox[1] + anchor.bbox[3] / 2
    nearby = [
        b for b in blocks
        if abs((b.bbox[1] + b.bbox[3] / 2) - ay) <= max_dy
    ]
    if not all(any(w in b.text.lower() for b in nearby) for w in words):
        return None

    remaining = set(words)
    selected: list[OCRBlock] = []
    while remaining:
        best = max(
            (b for b in nearby if b not in selected),
            key=lambda b: sum(1 for w in remaining if w in b.text.lower()),
            default=None,
        )
        if best is None:
            break
        gained = {w for w in remaining if w in best.text.lower()}
        if not gained:
            break
        selected.append(best)
        remaining -= gained
    return selected if not remaining else None


def _parse_abv_pct(field: ExtractedField | None) -> float | None:
    if field is None or not field.value:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)", field.value.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
