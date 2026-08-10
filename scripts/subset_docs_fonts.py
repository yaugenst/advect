# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools[woff]>=4.55"]
# ///
"""Regenerate the Braille-only Adwaita Mono webfont used by the docs.

The three Source Code Pro files are unmodified upstream release assets. Their
pinned URLs and checksums live in ``docs-theme/THIRD_PARTY_LICENSES``.

Run from the repository root:

    uv run --script scripts/subset_docs_fonts.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs-theme" / "fonts"

BRAILLE_UNICODES = "2800-28FF"
SOURCE = Path("/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf")
LICENSE = Path("/usr/share/licenses/adwaita-mono-fonts/LICENSE")


def main() -> None:
    """Subset the Braille face and copy its license text."""
    pyftsubset = shutil.which("pyftsubset")
    if pyftsubset is None:
        message = "pyftsubset not found (install fonttools)"
        raise SystemExit(message)
    if not SOURCE.exists():
        message = f"missing source font: {SOURCE}"
        raise SystemExit(message)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "braille-mono.woff2"
    subprocess.run(  # noqa: S603 - fixed local font tooling
        [
            pyftsubset,
            str(SOURCE),
            f"--unicodes={BRAILLE_UNICODES}",
            "--flavor=woff2",
            f"--output-file={out}",
        ],
        check=True,
    )
    print(f"{out.name}: {out.stat().st_size // 1024} KB (from {SOURCE.name})")
    shutil.copyfile(LICENSE, OUT / "LICENSE-adwaita-mono")
    print("LICENSE-adwaita-mono: copied")


if __name__ == "__main__":
    main()
