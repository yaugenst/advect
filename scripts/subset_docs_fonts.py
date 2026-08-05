# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools[woff]>=4.55"]
# ///
"""Regenerate the subset webfonts committed under docs-theme/fonts/.

The docs theme ships a subset of Source Code Pro. Fonts are subset from locally
installed system fonts, so the emitted files depend on the installed font
versions; the results are committed because contributors should not need these
exact fonts installed to build the docs.

Run from the repository root:

    uv run --script scripts/subset_docs_fonts.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs-theme" / "fonts"

# Basic/extended latin, punctuation, sub/superscripts, greek, arrows, math
# operators, misc technical, box drawing, blocks, geometric shapes, dingbats.
TEXT_UNICODES = (
    "0020-007E,00A0-017F,02C6-02DC,0370-03FF,2000-206F,2070-209F,2100-214F,"
    "2190-21FF,2200-22FF,2300-23FF,2500-257F,2580-259F,25A0-25FF,2700-27BF"
)
BRAILLE_UNICODES = "2800-28FF"

FACES = [
    # (output stem, source, unicodes)
    (
        "scp-regular",
        "/usr/share/fonts/adobe-source-code-pro-fonts/SourceCodePro-Regular.otf",
        TEXT_UNICODES,
    ),
    (
        "scp-italic",
        "/usr/share/fonts/adobe-source-code-pro-fonts/SourceCodePro-It.otf",
        TEXT_UNICODES,
    ),
    (
        "scp-bold",
        "/usr/share/fonts/adobe-source-code-pro-fonts/SourceCodePro-Bold.otf",
        TEXT_UNICODES,
    ),
    # Braille block only: the playground's dot-matrix plots and logo. Without
    # it those glyphs fall back to whatever monospace the OS has, with
    # mismatched metrics that misalign the stacked plot layers.
    (
        "braille-mono",
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        BRAILLE_UNICODES,
    ),
]

LICENSES = {
    "LICENSE-source-code-pro.md": "/usr/share/licenses/adobe-source-code-pro-fonts/LICENSE.md",
    "LICENSE-adwaita-mono": "/usr/share/licenses/adwaita-mono-fonts/LICENSE",
}


def main() -> None:
    """Subset each face and copy the license texts."""
    pyftsubset = shutil.which("pyftsubset")
    if pyftsubset is None:
        message = "pyftsubset not found (install fonttools)"
        raise SystemExit(message)
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, src, unicodes in FACES:
        source = Path(src)
        if not source.exists():
            message = f"missing source font: {source}"
            raise SystemExit(message)
        out = OUT / f"{stem}.woff2"
        subprocess.run(  # noqa: S603 - fixed local font tooling
            [
                pyftsubset,
                str(source),
                f"--unicodes={unicodes}",
                "--flavor=woff2",
                f"--output-file={out}",
            ],
            check=True,
        )
        print(f"{out.name}: {out.stat().st_size // 1024} KB (from {source.name})")
    for name, src in LICENSES.items():
        shutil.copyfile(src, OUT / name)
        print(f"{name}: copied")


if __name__ == "__main__":
    main()
