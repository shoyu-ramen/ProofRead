# Sources & licensing — real_labels starter set

Every image in this directory must have a one-sentence provenance entry
here. If you can't write one truthfully, don't commit the file.

## Licensing posture for the starter set

All starter-set images are sourced from the **TTB Public COLA Registry**
(`ttbonline.gov/colasonline`), the public regulatory record of approved
Certificate of Label Approval / Exemption (TTB Form 5100.31) submissions.
The user-provided sourcing brief explicitly authorizes this category as
allowed — "(b) a public regulatory record" — and TTB compliance is the
exact regime ProofRead validates against.

Caveats the next reviewer should hold in mind:

- The artwork itself is brewer-submitted creative work and remains the
  brewer's copyright. TTB hosts it as part of the public regulatory
  record. We are using these images for a non-commercial validation
  corpus, not redistributing them as artwork.
- COLA artwork is the **unrolled label design** as submitted, not a
  photograph of a physical container. `front.jpg` and `back.jpg` for
  every starter-set entry are duplicates of the same composite — the
  COLA submission is a single file. A future contribution layer that
  adds real bottle/can photos under SPEC §0.5 conditions (sunlight,
  curved bottle, condensation, etc.) is still needed; this starter set
  only satisfies the "label content is real" half of the requirement,
  not the "capture conditions are real" half.
- Health Warning ground truth (`beer.health_warning.exact_text`) was
  marked `pass` based on visual transcription from a screen render of
  the COLA artwork. The whole point of this corpus is to surface OCR
  vs. canonical-text diffs in the ≤1-character-edit-distance regime;
  if a real reviewer with the original COLA PDF spots a typo I missed,
  flip the ground-truth value and add a note here.

## Retrieval

Images were retrieved with `curl` against the public registry on
2026-04-28. Reproduction recipe (from any browser/terminal):

1. Visit `https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do`
2. Search by Class/Type code 901 (BEER) and a date range.
3. Click a result, then "Printable Version" to reach `viewColaDetails.do?action=publicFormDisplay&ttbid=<id>`.
4. The artwork attachment URL is in that page as
   `/colasonline/publicViewAttachment.do?filename=<file>&filetype=l`.

The session-priming step (basic detail → form-display → attachment in
the same JSESSIONID) is required; fetching the attachment URL without
having visited the form-display first returns an "Unable to render
attachment" error page.

## Per-label index

| ID         | TTB ID         | Brewer                      | Brand                         | Source URL |
|------------|----------------|-----------------------------|-------------------------------|------------|
| lbl-0001   | 23164001000869 | Catskill Brewery, LLC       | BARKABOOM Oktoberfest Lager   | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=23164001000869 |
| lbl-0002   | 23177001000878 | Brew Hub, LLC               | Ultra Right                   | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=23177001000878 |
| lbl-0003   | 22179001000295 | Calvert Brewing Company, LLC| German Pilsner                | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=22179001000295 |
| lbl-0004   | 19163001000306 | Calvert Brewing Company, LLC| Autumn Oktoberfest            | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=19163001000306 |
| lbl-0005   | 19122001000872 | Calvert Brewing Company, LLC| 7th State Golden Lager        | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=19122001000872 |
| lbl-0006   | 20072001000036 | Calvert Brewing Company, LLC| 7th State Golden Lager (rev.) | https://www.ttbonline.gov/colasonline/viewColaDetails.do?action=publicFormDisplay&ttbid=20072001000036 |

## Per-label provenance (one sentence each)

- **lbl-0001** — BARKABOOM Oktoberfest Lager, TTB ID 23164001000869,
  approved 2023-06-14 (status: SURRENDERED). Public regulatory record
  retrieved from TTB COLA Public Registry on 2026-04-28; original COLA
  attachment filename `CB_BarkABoom_16OZ_label.jpg`, 2368×1564 RGB JPEG.
- **lbl-0002** — "Conservative Dad's Ultra Right", TTB ID 23177001000878,
  approved 2023-07-13 (status: EXPIRED). Public regulatory record
  retrieved from TTB COLA Public Registry on 2026-04-28; original COLA
  attachment filename `tcm-12oz-can.jpg`, 2895×1609 RGB JPEG.
- **lbl-0003** — Calvert Brewing German Pilsner, TTB ID 22179001000295,
  approved 2022-06-30 (status: SURRENDERED). Public regulatory record
  retrieved from TTB COLA Public Registry on 2026-04-28; original COLA
  attachment filename `germanpilscan.png`, 1093×692 RGBA PNG (re-saved
  as JPEG with white matte for the corpus).
- **lbl-0004** — Calvert Brewing Autumn Oktoberfest (Märzen), TTB ID
  19163001000306, approved 2019-07-01 (status: SURRENDERED). Public
  regulatory record retrieved from TTB COLA Public Registry on
  2026-04-28; original COLA attachment filename `Oktoberfest.jpg`,
  1536×685 RGB JPEG.
- **lbl-0005** — Calvert Brewing 7th State Golden Lager (16oz), TTB ID
  19122001000872, approved 2019-05-18 (status: SURRENDERED). Public
  regulatory record retrieved from TTB COLA Public Registry on
  2026-04-28; original COLA attachment filename `7th State 16 oz.jpg`,
  1189×740 RGB JPEG.
- **lbl-0006** — Calvert Brewing 7th State Golden Lager (revised
  artwork), TTB ID 20072001000036, approved 2020-03-24 (status:
  SURRENDERED). Public regulatory record retrieved from TTB COLA
  Public Registry on 2026-04-28; original COLA attachment filename
  `7th State TTB.jpg`, 1304×577 RGB JPEG.

## What was NOT included and why

- **TTB ID 22179001000289** ("German Style Pilsner", Calvert Brewing,
  approved 2022-06-30): downloaded and verified as a real beer COLA
  attachment (`Germanpilsner.png`, 776×717 RGBA PNG) but **excluded**
  because both edges are < 1024 px and the README requires the long
  edge ≥ 1024 px. Upscaling would not add OCR signal and would
  misrepresent capture realism.
- **Wikimedia Commons** beer/wine/spirit label searches were **not
  exhausted** in this pass. They are the right next source for
  *photograph-of-real-bottle* coverage (where COLA artwork is weakest);
  see "Recommended next steps" below. Each Commons hit must be license-
  audited individually before commit.
- **Untappd / BeerAdvocate / RateBeer / Vivino / brewery sites /
  Instagram / Reddit / Flickr-without-CC**: untouched per the brief.

## Recommended next steps

1. Add 5–10 Wikimedia Commons photographs of real beer cans/bottles
   under CC-BY / CC0 / PD licenses. These complement the COLA
   artwork by adding the photographic-capture noise (specular
   highlights, curvature distortion, lighting variation) that
   COLA submissions strip away. Audit each license individually,
   record the Commons file URL + license tag in this file, and
   transcribe `label_spec` from what is actually visible.
2. When that 2nd pass lands, drop the per-COLA `front.jpg ==
   back.jpg` duplication note from the README and adopt the
   front/back split for true photographs.
3. Once a `validation/real_corpus.py` loader exists, rerun the
   active extractor against this corpus and reconcile any
   `health_warning.exact_text` mismatches against the original
   COLA PDFs — that's the single most valuable reconciliation
   pass for SPEC v1.13's 500-label test-set goal.
