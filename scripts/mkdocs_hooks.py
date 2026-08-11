"""Prepare and validate generated documentation assets."""

from __future__ import annotations

import hashlib
import json
import os
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
_RAW_MKDOCSTRINGS_DIRECTIVE = re.compile(r"^:::\s+advect(?:[.\s]|$)", re.MULTILINE)
_RAW_SPHINX_ROLE = re.compile(r":(?:attr|class|data|exc|func|meth|mod):")
_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _unescaped_pipe_count(line: str) -> int:
    """Count Markdown table separators, excluding escaped literal pipes."""
    return sum(
        character == "|" and (index == 0 or line[index - 1] != "\\")
        for index, character in enumerate(line)
    )


def _malformed_markdown_table_lines(markdown: str) -> tuple[int, ...]:
    """Return rows whose cell count disagrees with their Markdown table header."""
    lines = markdown.splitlines()
    malformed: list[int] = []
    for index, line in enumerate(lines[:-1]):
        if not (
            _MARKDOWN_TABLE_ROW.fullmatch(line)
            and _MARKDOWN_TABLE_SEPARATOR.fullmatch(lines[index + 1])
        ):
            continue
        expected_pipes = _unescaped_pipe_count(line)
        for row_index in range(index + 2, len(lines)):
            row = lines[row_index]
            if not _MARKDOWN_TABLE_ROW.fullmatch(row):
                break
            if _unescaped_pipe_count(row) != expected_pipes:
                malformed.append(row_index + 1)
    return tuple(malformed)


def _api_rendering_errors(page: str, markdown: str) -> tuple[str, ...]:
    """Describe unrendered or structurally invalid generated API markup."""
    errors: list[str] = []
    if _RAW_MKDOCSTRINGS_DIRECTIVE.search(markdown):
        errors.append(f"{page}: unexpanded mkdocstrings directive")
    if _RAW_SPHINX_ROLE.search(markdown):
        errors.append(f"{page}: raw Sphinx role")
    malformed_rows = _malformed_markdown_table_lines(markdown)
    if malformed_rows:
        rendered_rows = ", ".join(str(line) for line in malformed_rows)
        errors.append(f"{page}: malformed Markdown table rows {rendered_rows}")
    return tuple(errors)


def _package_version() -> str:
    cargo = tomllib.loads((_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return str(cargo["workspace"]["package"]["version"])


def on_config(config: MkDocsConfig) -> None:
    """Give the theme stylesheet a content-derived browser cache key."""
    stylesheet = _ROOT / "docs-theme" / "css" / "theme.css"
    config.extra["theme_css_version"] = hashlib.sha256(stylesheet.read_bytes()).hexdigest()[:12]


def _stage_browser_assets(site_dir: Path) -> None:
    """Copy the playground adapter and current browser wheel into the built site."""
    assets = site_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        _ROOT / "docs-theme" / "playground_runtime.py",
        assets / "playground_runtime.py",
    )

    wheel_pattern = f"advect-{_package_version()}-*.whl"
    wheels = list((_ROOT / "dist" / "pyodide").glob(wheel_pattern))
    if not wheels:
        if not os.environ.get("ADVECT_REQUIRE_BROWSER_WHEEL"):
            return
        message = (
            "no browser wheel: run `mkdir -p dist && "
            "uvx --from pyodide-build==0.36.0 pyodide build . "
            "--outdir dist/pyodide` first"
        )
        raise FileNotFoundError(message)
    if len(wheels) != 1:
        message = f"expected one current browser wheel, found {len(wheels)}"
        raise RuntimeError(message)
    wheel = wheels[0]
    shutil.copyfile(wheel, assets / wheel.name)

    (assets / "advect-browser-wheel.json").write_text(
        json.dumps({"filename": wheel.name}, indent=2) + "\n",
        encoding="utf-8",
    )


@event_priority(-100)
def on_post_build(config: MkDocsConfig) -> None:
    """Stage browser assets and validate agent documentation after plugins finish."""
    site_dir = Path(config.site_dir)
    _stage_browser_assets(site_dir)
    site_url = str(config.site_url).rstrip("/") + "/"
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    llms_text = (site_dir / "llms.txt").read_text(encoding="utf-8")

    landing_contract = (
        f'<link rel="canonical" href="{site_url}">',
        (
            '<link rel="alternate" type="text/plain" '
            'title="Advect documentation for AI agents" href="llms.txt">'
        ),
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

    api_markdown: dict[str, str] = {}
    for url in markdown_urls:
        relative = Path(unquote(url.removeprefix(site_url)))
        if relative.parts[:1] != ("api",):
            continue
        page = relative.parent.relative_to("api").as_posix()
        if page == ".":
            page = "index"
        api_markdown[page] = (site_dir / relative).read_text(encoding="utf-8")
    if not api_markdown:
        message = "llms.txt does not advertise any generated API Markdown"
        raise RuntimeError(message)

    rendering_errors: list[str] = []
    for page, api_text in api_markdown.items():
        rendering_errors.extend(_api_rendering_errors(page, api_text))
    if rendering_errors:
        message = "generated API Markdown contains rendering artifacts:\n" + "\n".join(
            rendering_errors
        )
        raise RuntimeError(message)

    api_contracts = {
        "arrays": ("## array", "## asarray", "## stop_gradient"),
        "errors": ("## AdvectError", "## ImplicitSolveError"),
        "primitives": ("## def_abstract", "## def_jvp", "## def_transpose"),
        "staging": ("## stage", "## StagedProgram", "## vjp_program"),
        "transforms": ("## grad", "Examples:"),
        "numpy": ("### array", "### asanyarray", "### asarray"),
        "pytree": ("### register_pytree_node", "### tree_flatten"),
        "testing": ("### check_gradient", "### check_primitive"),
        "support": ("## support_catalog",),
        "scipy": ("Special functions", "Image processing", "Solver callbacks"),
        "scipy/special": ("### gammaln", "### log_softmax"),
        "scipy/ndimage": ("### gaussian_filter", "### black_tophat"),
        "scipy/solvers": ("### root_solver", "### gmres_solver"),
        "interop": ("All three bridges are first-order reverse-mode boundaries",),
        "interop/torch": ("first-order PyTorch operation",),
        "interop/jax": ("first-order JAX operation",),
        "interop/autograd": ("first-order HIPS Autograd primitive",),
    }
    for page, markers in api_contracts.items():
        api_text = api_markdown.get(page, "")
        if not all(marker in api_text for marker in markers):
            message = (
                f"generated {page!r} API Markdown does not contain its expanded public docstrings"
            )
            raise RuntimeError(message)
