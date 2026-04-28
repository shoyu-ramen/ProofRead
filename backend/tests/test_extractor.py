"""Beer field extraction tests."""

from app.services.extractors.beer import extract_beer_fields
from app.services.ocr import OCRBlock, OCRResult


def make_ocr(text: str) -> OCRResult:
    blocks = [
        OCRBlock(text=line, bbox=(0, i * 30, 400, 25), confidence=0.99)
        for i, line in enumerate(text.split("\n"))
    ]
    return OCRResult(full_text=text, blocks=blocks, provider="test", raw={})


def test_extracts_health_warning(canonical_warning):
    text = "ANYTOWN ALE\n5.5% ABV\n12 FL OZ\n" + canonical_warning
    ctx = extract_beer_fields({"back": make_ocr(text)}, container_size_ml=355)
    hw = ctx.fields.get("health_warning")
    assert hw is not None
    assert hw.value is not None
    assert "GOVERNMENT WARNING" in hw.value
    assert "Surgeon General" in hw.value
    assert "health problems" in hw.value


def test_extracts_abv():
    ctx = extract_beer_fields({"front": make_ocr("ANYTOWN ALE\n5.5% ABV")}, container_size_ml=355)
    assert ctx.fields["alcohol_content"].value is not None
    assert "5.5" in ctx.fields["alcohol_content"].value
    assert ctx.abv_pct == 5.5


def test_extracts_net_contents():
    ctx = extract_beer_fields({"front": make_ocr("ANYTOWN ALE\n12 FL OZ")}, container_size_ml=355)
    val = ctx.fields["net_contents"].value
    assert val is not None
    assert "12" in val


def test_extracts_class_type_prefers_specific():
    """India Pale Ale should match before generic Ale."""
    ctx = extract_beer_fields(
        {"front": make_ocr("ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV")},
        container_size_ml=355,
    )
    assert ctx.fields["class_type"].value == "india pale ale"


def test_extracts_brand_name_from_largest_front_block():
    front = OCRResult(
        full_text="HUGE BRAND\nipa\n5.5% abv",
        blocks=[
            OCRBlock(text="HUGE BRAND", bbox=(0, 0, 600, 100)),
            OCRBlock(text="ipa", bbox=(0, 120, 100, 30)),
            OCRBlock(text="5.5% abv", bbox=(0, 160, 100, 30)),
        ],
        provider="test",
    )
    ctx = extract_beer_fields({"front": front}, container_size_ml=355)
    assert ctx.fields["brand_name"].value == "HUGE BRAND"


def test_country_of_origin_only_when_imported():
    text = "ANYTOWN ALE\n5.5% ABV\nProduct of Germany"
    ctx_dom = extract_beer_fields(
        {"front": make_ocr(text)}, container_size_ml=355, is_imported=False
    )
    ctx_imp = extract_beer_fields(
        {"front": make_ocr(text)}, container_size_ml=355, is_imported=True
    )
    assert "country_of_origin" not in ctx_dom.fields
    assert ctx_imp.fields["country_of_origin"].value is not None
    assert "germany" in ctx_imp.fields["country_of_origin"].value.lower()


def test_extracts_name_address():
    text = "ANYTOWN ALE\nBrewed and bottled by Anytown Brewing Co., Anytown, ST"
    ctx = extract_beer_fields({"back": make_ocr(text)}, container_size_ml=355)
    val = ctx.fields["name_address"].value
    assert val is not None
    assert "Anytown Brewing Co" in val


# --- Health Warning extraction edge cases (review issues #2 and #3) ---


def test_health_warning_trims_at_terminal_phrase(canonical_warning):
    """Trailing label text after the warning must not pollute the snippet."""
    text = (
        canonical_warning
        + " Please recycle this can responsibly. Visit example.com."
    )
    ctx = extract_beer_fields({"back": make_ocr(text)}, container_size_ml=355)
    hw = ctx.fields["health_warning"].value
    assert hw is not None
    assert hw.rstrip().endswith("health problems.")
    assert "recycle" not in hw.lower()


def test_health_warning_handles_warning_at_end_of_text(canonical_warning):
    """Warning is the last text on the label — common back-label layout."""
    text = "Some preamble.\n\n" + canonical_warning
    ctx = extract_beer_fields({"back": make_ocr(text)}, container_size_ml=355)
    hw = ctx.fields["health_warning"].value
    assert hw is not None
    assert "GOVERNMENT WARNING" in hw
    assert hw.rstrip().endswith("health problems.")


def test_fuzzy_anchor_finds_warning_with_one_letter_dropped(canonical_warning):
    """'GOVERMENT WARNING' (missing N) — fuzzy anchor still locates the body."""
    typoed = canonical_warning.replace("GOVERNMENT WARNING", "GOVERMENT WARNING")
    ctx = extract_beer_fields({"back": make_ocr(typoed)}, container_size_ml=355)
    assert "health_warning" in ctx.fields
    hw = ctx.fields["health_warning"].value
    assert "Surgeon General" in hw
    assert "health problems" in hw


def test_fuzzy_anchor_finds_warning_with_letter_dropped_in_warning_word(canonical_warning):
    """'GOVERNMENT WARNIG' — missing G in WARNING."""
    typoed = canonical_warning.replace("GOVERNMENT WARNING", "GOVERNMENT WARNIG")
    ctx = extract_beer_fields({"back": make_ocr(typoed)}, container_size_ml=355)
    assert "health_warning" in ctx.fields


def test_anchor_not_found_returns_none():
    """Random text without anything resembling the anchor → no field extracted."""
    text = "ANYTOWN ALE\n5.5% ABV\n12 FL OZ\nEnjoy responsibly."
    ctx = extract_beer_fields({"back": make_ocr(text)}, container_size_ml=355)
    assert "health_warning" not in ctx.fields


# --- _bbox_for_text spatial-cluster fix (review issue #6) ---


def test_bbox_unions_adjacent_split_anchor_blocks():
    """Anchor split across two adjacent OCR blocks → union them legitimately."""
    from app.services.extractors.beer import _bbox_for_text

    blocks = [
        OCRBlock(text="GOVERNMENT", bbox=(50, 800, 200, 30)),
        OCRBlock(text="WARNING:", bbox=(260, 800, 100, 30)),
    ]
    bbox = _bbox_for_text(blocks, "GOVERNMENT WARNING")
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 50
    assert y == 800
    assert w >= 310, f"bbox should span both blocks; got width={w}"


def test_bbox_excludes_distant_block_sharing_only_one_word():
    """A 'WARNING' word in another label region must not be unioned in.
    Without spatial clustering, the bbox would span from y=200 to y=830 — an
    obviously-wrong highlight covering two unrelated label regions.
    """
    from app.services.extractors.beer import _bbox_for_text

    blocks = [
        OCRBlock(text="GOVERNMENT", bbox=(50, 800, 200, 30)),
        OCRBlock(text="WARNING:", bbox=(260, 800, 100, 30)),
        OCRBlock(text="WARNING: keep refrigerated", bbox=(50, 200, 350, 30)),
    ]
    bbox = _bbox_for_text(blocks, "GOVERNMENT WARNING")
    assert bbox is not None
    _, y, _, h = bbox
    assert y >= 700, f"bbox includes distant unrelated block; y={y}, h={h}"
    assert y + h <= 850, f"bbox extends past the warning region; y+h={y + h}"


def test_bbox_returns_none_when_required_word_absent():
    """If 'GOVERNMENT' is nowhere in the OCR but 'WARNING' appears, the
    fallback used to highlight the WARNING block alone — wrong. Now returns
    None so the report doesn't show a misleading highlight."""
    from app.services.extractors.beer import _bbox_for_text

    blocks = [
        OCRBlock(text="STORAGE WARNING: refrigerate", bbox=(50, 200, 350, 30)),
        OCRBlock(text="ANYTOWN ALE", bbox=(50, 100, 200, 60)),
    ]
    bbox = _bbox_for_text(blocks, "GOVERNMENT WARNING")
    assert bbox is None


def test_bbox_strict_substring_path_unaffected():
    """A single block fully containing the snippet → strict path returns its bbox."""
    from app.services.extractors.beer import _bbox_for_text

    blocks = [
        OCRBlock(text="GOVERNMENT WARNING: blah blah", bbox=(50, 800, 600, 30)),
    ]
    bbox = _bbox_for_text(blocks, "GOVERNMENT WARNING")
    assert bbox == (50, 800, 600, 30)
