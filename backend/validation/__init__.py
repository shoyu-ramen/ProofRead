"""ProofRead OCR / vision-extractor validation harness.

Synthesizes labeled beer-label fixtures, runs them through the production
pipeline (`app.services.pipeline.process_scan`) under a configurable OCR
provider, and reports precision/recall/F1 per rule_id.

The harness is extractor-agnostic: it accepts any OCRProvider (the mock
fixture from `app.services.ocr`, the stub Google Vision provider, or a
future Claude-vision provider). The default is a "perfect mock" that
returns the synthesizer's ground-truth text directly. That validates the
*harness plumbing*, not real OCR; real OCR validation is gated behind the
`real_ocr` pytest mark.

See `validation/README.md` for methodology.
"""
