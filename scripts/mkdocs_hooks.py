"""Prepare and validate generated documentation assets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from mkdocs.plugins import event_priority

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

_ROOT = Path(__file__).parent.parent
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+\.md)\)")


def _package_version() -> str:
    cargo = tomllib.loads((_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return str(cargo["workspace"]["package"]["version"])


def on_config(config: MkDocsConfig) -> None:
    """Give the theme stylesheet a content-derived browser cache key."""
    stylesheet = _ROOT / "docs-theme" / "css" / "theme.css"
    config.extra["theme_css_version"] = hashlib.sha256(stylesheet.read_bytes()).hexdigest()[:12]


def on_pre_build(config: object) -> None:
    """Stage the playground adapter and the browser wheel."""
    del config
    assets = _ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        _ROOT / "docs-theme" / "playground_runtime.py",
        assets / "playground_runtime.py",
    )

    # the pyodide wheel: prefer a fresh build, else keep the staged copy
    wheel_pattern = f"advect-{_package_version()}-*.whl"
    wheels = list((_ROOT / "dist" / "pyodide").glob(wheel_pattern))
    staged_wheels = list(assets.glob(wheel_pattern))
    if wheels:
        if len(wheels) != 1:
            message = f"expected one current browser wheel, found {len(wheels)}"
            raise RuntimeError(message)
        wheel = wheels[0]
        for stale_wheel in assets.glob("advect-*.whl"):
            stale_wheel.unlink()
        shutil.copyfile(wheel, assets / wheel.name)
    elif not staged_wheels:
        message = (
            "no browser wheel: run `mkdir -p dist && "
            "uvx --from pyodide-build==0.36.0 pyodide build . "
            "--outdir dist/pyodide` first"
        )
        raise FileNotFoundError(message)
    else:
        if len(staged_wheels) != 1:
            message = f"expected one staged current browser wheel, found {len(staged_wheels)}"
            raise RuntimeError(message)
        wheel = staged_wheels[0]

    (assets / "advect-browser-wheel.json").write_text(
        json.dumps({"filename": wheel.name}, indent=2) + "\n",
        encoding="utf-8",
    )


@event_priority(-100)
def on_post_build(config: MkDocsConfig) -> None:
    """Validate the version-local agent documentation after plugins finish."""
    site_dir = Path(config.site_dir)
    site_url = str(config.site_url).rstrip("/") + "/"
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    llms_text = (site_dir / "llms.txt").read_text(encoding="utf-8")

    landing_contract = (
        f'<link rel="canonical" href="{site_url}">',
        '<link rel="alternate" type="text/plain" '
        'title="Advect documentation for AI agents" href="llms.txt">',
        '<section class="agent-context" aria-label="Machine-readable documentation">',
        '<a href="llms.txt">llms.txt</a>',
    )
    if not all(fragment in index_html for fragment in landing_contract):
        message = "landing page does not advertise its version-local agent documentation"
        raise RuntimeError(message)

    markdown_urls = _MARKDOWN_LINK.findall(llms_text)
    if not markdown_urls:
        message = "llms.txt does not advertise any Markdown documents"
        raise RuntimeError(message)
    missing = [
        url
        for url in markdown_urls
        if not url.startswith(site_url)
        or not (site_dir / unquote(url.removeprefix(site_url))).is_file()
    ]
    if missing:
        message = "llms.txt advertises files outside this version or build:\n" + "\n".join(missing)
        raise RuntimeError(message)

    api_text = (site_dir / "api" / "transforms" / "index.md").read_text(encoding="utf-8")
    if "## grad" not in api_text or "Examples:" not in api_text or "::: advect" in api_text:
        message = "generated API Markdown does not contain expanded docstrings and examples"
        raise RuntimeError(message)
