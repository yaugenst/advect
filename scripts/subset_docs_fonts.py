# /// script
# requires-python = ">=3.12"
# dependencies = ["brotli==1.2.0", "fonttools[woff]==4.63.0"]
# ///
"""Regenerate the Braille-only Adwaita Mono webfont used by the docs.

The pinned upstream version, source URLs, and checksums live in
``docs-theme/THIRD_PARTY_LICENSES``.

Run from the repository root:

    uv run --script scripts/subset_docs_fonts.py
"""

from __future__ import annotations

import tempfile
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

from fontTools.subset import Options, Subsetter, load_font, save_font

OUT = Path(__file__).parent.parent / "docs-theme" / "fonts"

BRAILLE_UNICODES = range(0x2800, 0x2900)
SOURCE_URL = "https://gitlab.gnome.org/GNOME/adwaita-fonts/-/raw/50.0/mono/AdwaitaMono-Regular.ttf"
SOURCE_SHA256 = "0edc6a8d8ca249f594dee661b2f57f1a3baa33bc7aca826c6a9fe27b06f9f930"
LICENSE_URL = "https://gitlab.gnome.org/GNOME/adwaita-fonts/-/raw/50.0/LICENSE"
LICENSE_SHA256 = "459687971d21c53923c1d1c9c062ec273a7ea03226b36195b79ec6af7d98dc81"
OUTPUT_SHA256 = "fdd8a403be78c201e149a8d60664112d7ebc21a4a1795d4a2ef22273a8a7912b"


def _download(url: str, path: Path, expected_sha256: str) -> None:
    """Download one pinned upstream file and verify its contents."""
    urlretrieve(url, path)  # noqa: S310 - fixed HTTPS URLs are checksum-verified
    actual_sha256 = sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        message = f"checksum mismatch for {url}: {actual_sha256}"
        raise SystemExit(message)


def main() -> None:
    """Download and subset the pinned Braille face and its license."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "braille-mono.woff2"
    license_out = OUT / "LICENSE-adwaita-mono"
    with tempfile.TemporaryDirectory(dir=OUT) as temporary_directory:
        source = Path(temporary_directory) / "AdwaitaMono-Regular.ttf"
        license_source = Path(temporary_directory) / "LICENSE"
        generated = Path(temporary_directory) / out.name
        _download(SOURCE_URL, source, SOURCE_SHA256)
        _download(LICENSE_URL, license_source, LICENSE_SHA256)
        options = Options()
        options.flavor = "woff2"
        font = load_font(str(source), options, lazy=False)
        subsetter = Subsetter(options=options)
        subsetter.populate(unicodes=BRAILLE_UNICODES)
        subsetter.subset(font)
        save_font(font, str(generated), options)

        output_sha256 = sha256(generated.read_bytes()).hexdigest()
        if output_sha256 != OUTPUT_SHA256:
            message = f"unexpected {out.name} checksum: {output_sha256}"
            raise SystemExit(message)
        generated.replace(out)
        license_source.replace(license_out)
    print(f"{out.name}: {out.stat().st_size // 1024} KB ({output_sha256})")
    print(f"{license_out.name}: {LICENSE_SHA256}")


if __name__ == "__main__":
    main()
