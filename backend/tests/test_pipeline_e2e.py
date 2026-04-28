"""End-to-end pipeline test: OCR (mocked) → extractor → rule engine → report."""

from app.rules.types import CheckOutcome
from app.services.ocr import MockOCRProvider
from app.services.pipeline import ScanInput, process_scan


def _ocr_fixture(text: str) -> dict:
    return {
        "full_text": text,
        "blocks": [
            {"text": line, "bbox": [0, i * 30, 400, 25], "confidence": 0.99}
            for i, line in enumerate(text.split("\n"))
        ],
    }


def _make_scan(
    images_text: dict[str, str], is_imported: bool = False
) -> tuple[ScanInput, MockOCRProvider]:
    # Mock provider that returns different fixtures per surface based on hint.
    fixtures = {surface: _ocr_fixture(t) for surface, t in images_text.items()}

    class _Multi:
        def process(self, image_bytes: bytes, hint: str | None = None):
            return MockOCRProvider(fixtures[hint]).process(image_bytes, hint)

    scan = ScanInput(
        beverage_type="beer",
        container_size_ml=355,
        images={s: f"image-bytes-{s}".encode() for s in images_text},
        is_imported=is_imported,
    )
    return scan, _Multi()


def test_compliant_label_has_no_failures(compliant_label_text):
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    scan, ocr = _make_scan({"front": front_text, "back": compliant_label_text})

    report = process_scan(scan, ocr, skip_capture_quality=True)

    failures = [r for r in report.rule_results if r.status == CheckOutcome.FAIL]
    assert not failures, f"Unexpected failures on compliant label: {failures}"
    assert report.overall in {"pass", "advisory"}


def test_label_with_typo_in_health_warning_fails(compliant_label_text):
    bad_text = compliant_label_text.replace("Surgeon", "Sergent")
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    scan, ocr = _make_scan({"front": front_text, "back": bad_text})

    report = process_scan(scan, ocr, skip_capture_quality=True)

    assert report.overall == "fail"
    hw = next(r for r in report.rule_results if r.rule_id == "beer.health_warning.exact_text")
    assert hw.status == CheckOutcome.FAIL
    assert hw.expected is not None
    assert hw.fix_suggestion is not None


def test_label_missing_health_warning_fails(compliant_label_text):
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back_text_no_warning = (
        "Brewed and bottled by Anytown Brewing Co., Anytown, ST 00000"
    )
    scan, ocr = _make_scan({"front": front_text, "back": back_text_no_warning})

    report = process_scan(scan, ocr, skip_capture_quality=True)

    hw = next(r for r in report.rule_results if r.rule_id == "beer.health_warning.exact_text")
    assert hw.status == CheckOutcome.FAIL


def test_imported_label_without_country_fails(compliant_label_text):
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    scan, ocr = _make_scan(
        {"front": front_text, "back": compliant_label_text},
        is_imported=True,
    )

    report = process_scan(scan, ocr, skip_capture_quality=True)

    coo = next(
        r for r in report.rule_results if r.rule_id == "beer.country_of_origin.presence_if_imported"
    )
    assert coo.status == CheckOutcome.FAIL


def test_imported_label_with_country_passes(compliant_label_text):
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = compliant_label_text + "\nProduct of Germany"
    scan, ocr = _make_scan({"front": front_text, "back": back}, is_imported=True)

    report = process_scan(scan, ocr, skip_capture_quality=True)

    coo = next(
        r for r in report.rule_results if r.rule_id == "beer.country_of_origin.presence_if_imported"
    )
    assert coo.status == CheckOutcome.PASS


def test_health_warning_advisory_size_does_not_fail_overall(compliant_label_text):
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    scan, ocr = _make_scan({"front": front_text, "back": compliant_label_text})

    report = process_scan(scan, ocr, skip_capture_quality=True)

    size_rule = next(r for r in report.rule_results if r.rule_id == "beer.health_warning.size")
    assert size_rule.status == CheckOutcome.ADVISORY


def test_label_with_trailing_recycling_text_still_passes(compliant_label_text):
    """End-to-end: trailing label text after the warning must not produce a false-fail."""
    front_text = "ANYTOWN ALE\nINDIA PALE ALE\n5.5% ABV\n12 FL OZ"
    back = compliant_label_text + "\nPlease recycle this can. Visit example.com."
    scan, ocr = _make_scan({"front": front_text, "back": back})

    report = process_scan(scan, ocr, skip_capture_quality=True)

    hw = next(r for r in report.rule_results if r.rule_id == "beer.health_warning.exact_text")
    assert hw.status == CheckOutcome.PASS, (
        f"HW should pass despite trailing text. Finding: {hw.finding}"
    )
