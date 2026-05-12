"""Wikimedia Commons fetcher with provenance + license enforcement.

Takes a Wikimedia file title or URL, queries the Commons MediaWiki API
for license + author + source metadata, downloads the image, and
populates a fresh `lbl-XXXX/` directory under
`validation/real_labels/`. Refuses anything not under an acceptable
license (CC0, CC-BY, CC-BY-SA, PD) so commit hygiene is enforced
upstream of the human reviewer.

Usage:

    # Fetch by canonical URL or by File:Title
    python -m validation.scripts.wikimedia_fetcher \\
        "https://commons.wikimedia.org/wiki/File:Example.jpg"

    python -m validation.scripts.wikimedia_fetcher "File:Example.jpg"

    # Pin to a specific item ID (overrides auto-increment)
    python -m validation.scripts.wikimedia_fetcher "File:Example.jpg" --id lbl-0042

    # Dry run — print metadata, do not write
    python -m validation.scripts.wikimedia_fetcher "File:Example.jpg" --dry-run

After this script writes `front.jpg` (and `back.jpg` as a duplicate
when no `--back` is given), run `validation/scripts/annotate.py` to
fill `truth.json`, then `record_extraction.py` to write the recording.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

_VALIDATION_ROOT = Path(__file__).resolve().parents[1]
_REAL_LABELS = _VALIDATION_ROOT / "real_labels"
_SOURCES_MD = _REAL_LABELS / "SOURCES.md"
_API = "https://commons.wikimedia.org/w/api.php"
_USER_AGENT = (
    "ProofRead-eval-fetcher/0.1 "
    "(https://github.com/shoyu-ramen/ProofRead; rosskuehl@gmail.com)"
)

# License substrings we accept. Wikimedia returns short codes that vary
# slightly by template ("CC BY 4.0" vs "Cc-by-4.0"); match
# case-insensitively on the substring so the matcher tolerates noise.
_ACCEPTED_LICENSE_PATTERNS = (
    "cc0",
    "cc-by",
    "cc by",
    "public domain",
    "pd-self",
    "pd-user",
)
_REJECTED_LICENSE_PATTERNS = (
    "non-commercial",
    "no derivatives",
    "fair use",
)


# ---------------------------------------------------------------------------
# Metadata + download
# ---------------------------------------------------------------------------


@dataclass
class CommonsMetadata:
    title: str  # "File:Example.jpg"
    image_url: str
    width: int
    height: int
    mime: str
    license_short: str
    license_url: str | None
    author_html: str | None
    description_url: str  # human page (commons.wikimedia.org/wiki/...)


def _normalise_title(arg: str) -> str:
    """Accept either `File:Foo.jpg` or any commons.wikimedia.org URL."""
    if arg.startswith("File:"):
        return arg
    parsed = urllib.parse.urlparse(arg)
    if "wikimedia.org" not in parsed.netloc:
        raise ValueError(
            f"unrecognised input {arg!r}; pass a `File:Foo.jpg` title or a "
            "commons.wikimedia.org URL"
        )
    # /wiki/File:Foo.jpg
    parts = parsed.path.split("/")
    for p in parts:
        if p.startswith("File:") or p.startswith("file:"):
            return urllib.parse.unquote(p[0].upper() + p[1:])
    raise ValueError(f"could not extract File: title from URL {arg!r}")


def _strip_html(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"<[^>]+>", "", text).strip()


def _is_accepted_license(license_short: str | None) -> bool:
    if not license_short:
        return False
    s = license_short.lower()
    if any(p in s for p in _REJECTED_LICENSE_PATTERNS):
        return False
    return any(p in s for p in _ACCEPTED_LICENSE_PATTERNS)


def fetch_metadata(title: str) -> CommonsMetadata:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "formatversion": "2",
    }
    response = httpx.get(
        _API, params=params, headers={"User-Agent": _USER_AGENT}, timeout=15.0
    )
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages") or []
    if not pages:
        raise RuntimeError(f"Wikimedia returned no pages for {title!r}")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"Wikimedia file does not exist: {title!r}")
    info_list = page.get("imageinfo") or []
    if not info_list:
        raise RuntimeError(f"Wikimedia returned no imageinfo for {title!r}")
    info = info_list[0]
    extmeta = info.get("extmetadata") or {}

    license_short = (extmeta.get("LicenseShortName") or {}).get("value")
    license_url = (extmeta.get("LicenseUrl") or {}).get("value")
    author_html = (extmeta.get("Artist") or {}).get("value")

    return CommonsMetadata(
        title=title,
        image_url=info["url"],
        width=int(info["width"]),
        height=int(info["height"]),
        mime=info["mime"],
        license_short=license_short or "(unknown)",
        license_url=license_url,
        author_html=author_html,
        description_url=info.get(
            "descriptionurl",
            f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
        ),
    )


def download_image(url: str) -> bytes:
    response = httpx.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Item placement
# ---------------------------------------------------------------------------


def _next_item_id() -> str:
    """Walk real_labels/ and pick the next free lbl-XXXX id.

    IDs are 4-digit padded; the script reserves 90xx for ad-hoc fixtures
    and avoids that range for new items so production-corpus IDs stay in
    a coherent block.
    """
    if not _REAL_LABELS.exists():
        return "lbl-0001"
    nums = [
        int(d.name.split("-")[1])
        for d in _REAL_LABELS.iterdir()
        if d.is_dir() and re.fullmatch(r"lbl-\d{4}", d.name)
    ]
    used = set(nums)
    candidate = max((n for n in nums if n < 9000), default=0) + 1
    while candidate in used:
        candidate += 1
    return f"lbl-{candidate:04d}"


def _validate_id(item_id: str) -> str:
    if not re.fullmatch(r"lbl-\d{4}", item_id):
        raise ValueError(f"invalid id {item_id!r}; expected lbl-XXXX")
    return item_id


# ---------------------------------------------------------------------------
# SOURCES.md upkeep
# ---------------------------------------------------------------------------


def _append_sources_entry(
    item_id: str, meta: CommonsMetadata, sha256_hex: str
) -> None:
    """Append a per-label provenance block to SOURCES.md.

    Keeps the existing file structure — the file already has a
    "Per-label provenance" section. Appends to that section as a new
    bullet so the diff is small and reviewers can see the new entry
    inline.
    """
    if not _SOURCES_MD.exists():
        # First-run safety: create a minimal SOURCES.md with just the
        # heading the appender expects. The seed file lives in tree
        # already, but a missing file shouldn't crash a fresh repo.
        _SOURCES_MD.write_text(
            "# Sources & licensing\n\n## Per-label provenance "
            "(one sentence each)\n\n"
        )
    text = _SOURCES_MD.read_text()
    bullet = (
        f"- **{item_id}** — {meta.title}, license: {meta.license_short}; "
        f"author: {_strip_html(meta.author_html) or '(unknown)'}; "
        f"source: {meta.description_url}; "
        f"original size: {meta.width}×{meta.height} {meta.mime}; "
        f"sha256: `{sha256_hex}`; fetched {date.today().isoformat()}.\n"
    )
    _SOURCES_MD.write_text(text.rstrip() + "\n" + bullet)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fetch(
    title_or_url: str,
    *,
    item_id: str | None,
    back_url: str | None,
    dry_run: bool,
) -> Path | None:
    title = _normalise_title(title_or_url)
    meta = fetch_metadata(title)

    print(f"title:    {meta.title}")
    print(f"license:  {meta.license_short}")
    print(f"author:   {_strip_html(meta.author_html) or '(unknown)'}")
    print(f"size:     {meta.width}×{meta.height} {meta.mime}")
    print(f"page:     {meta.description_url}")
    print(f"image:    {meta.image_url}")

    if not _is_accepted_license(meta.license_short):
        raise RuntimeError(
            f"refusing to fetch: license {meta.license_short!r} not in the "
            f"accepted set ({_ACCEPTED_LICENSE_PATTERNS})"
        )
    if max(meta.width, meta.height) < 1024:
        raise RuntimeError(
            f"refusing to fetch: long edge {max(meta.width, meta.height)}px "
            "< 1024px floor (linter would reject)"
        )

    if dry_run:
        print("[dry-run] not writing files")
        return None

    chosen_id = _validate_id(item_id) if item_id else _next_item_id()
    item_dir = _REAL_LABELS / chosen_id
    if item_dir.exists() and any(item_dir.iterdir()):
        raise RuntimeError(
            f"refusing to write into non-empty {item_dir}; pick a fresh "
            "--id or rm the directory first"
        )
    item_dir.mkdir(parents=True, exist_ok=True)

    front_bytes = download_image(meta.image_url)
    (item_dir / "front.jpg").write_bytes(front_bytes)

    if back_url:
        back_bytes = download_image(back_url)
        (item_dir / "back.jpg").write_bytes(back_bytes)
    else:
        # Most Wikimedia bottle photos show only one face. Duplicate to
        # back.jpg so the loader's two-file requirement is satisfied; the
        # annotator should note in capture_conditions that no real back is
        # available so an "advisory on health-warning" outcome is honest.
        (item_dir / "back.jpg").write_bytes(front_bytes)

    import hashlib

    sha = hashlib.sha256(front_bytes).hexdigest()
    _append_sources_entry(chosen_id, meta, sha)
    print(f"\nwrote {item_dir}/front.jpg + back.jpg")
    print(f"appended provenance to {_SOURCES_MD.name}")
    print(
        f"\nnext: python -m validation.scripts.annotate "
        f"validation/real_labels/{chosen_id}"
    )
    return item_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Wikimedia Commons image into validation/real_labels/."
    )
    parser.add_argument(
        "input",
        help="File:Foo.jpg title or full commons.wikimedia.org URL.",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Pin to a specific item id (e.g. lbl-0042). Auto-increments if omitted.",
    )
    parser.add_argument(
        "--back",
        default=None,
        help=(
            "Optional second file URL/title for back.jpg. When omitted, "
            "front.jpg is duplicated to back.jpg with an explicit note "
            "in the annotator workflow."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metadata without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        fetch(
            args.input,
            item_id=args.id,
            back_url=args.back,
            dry_run=args.dry_run,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
