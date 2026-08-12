"""Test generated-documentation validation helpers."""

from __future__ import annotations

import json
import runpy
from typing import TYPE_CHECKING

import pytest
from scripts import mkdocs_hooks
from scripts.mkdocs_hooks import _api_rendering_errors, _malformed_markdown_table_lines

if TYPE_CHECKING:
    from pathlib import Path


def test_generated_markdown_rejects_union_that_splits_a_table_cell() -> None:
    markdown = """\
| Name | Type | Description |
| --- | --- | --- |
| `value` | \\`str | None\\` | Optional value. |
"""

    assert _malformed_markdown_table_lines(markdown) == (3,)


def test_generated_markdown_accepts_lists_and_escaped_table_pipes() -> None:
    markdown = """\
- **`value`** (`str | None`, default: `None`) - Optional value.

| Name | Description |
| --- | --- |
| `value` | A literal \\| is escaped. |
"""

    assert _malformed_markdown_table_lines(markdown) == ()


def test_generated_api_markdown_rejects_every_rendering_artifact() -> None:
    markdown = """\
::: advect.grad

Returned by :func:`advect.vjp`.

| Name | Type |
| --- | --- |
| `value` | \\`str | None\\` |
"""

    assert _api_rendering_errors("transforms", markdown) == (
        "transforms: unexpanded mkdocstrings directive",
        "transforms: raw Sphinx role",
        "transforms: malformed Markdown table rows 7",
    )


def test_browser_wheel_is_required_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mkdocs_hooks, "_ROOT", tmp_path)
    monkeypatch.setattr(mkdocs_hooks, "_package_version", lambda: "0.1.0")
    monkeypatch.delenv("ADVECT_REQUIRE_BROWSER_WHEEL", raising=False)
    adapter = tmp_path / "docs-theme" / "playground_runtime.py"
    adapter.parent.mkdir(parents=True)
    adapter.touch()

    mkdocs_hooks._stage_browser_assets(tmp_path / "site")

    monkeypatch.setenv("ADVECT_REQUIRE_BROWSER_WHEEL", "1")
    with pytest.raises(FileNotFoundError, match="no browser wheel"):
        mkdocs_hooks._stage_browser_assets(tmp_path / "site")


def test_playground_accepts_integer_plot_boundary() -> None:
    runtime = runpy.run_path(str(mkdocs_hooks._ROOT / "docs-theme" / "playground_runtime.py"))

    runtime["playground_trace_json"]("x")

    assert json.loads(runtime["playground_evaluate_json"](4)) == [4.0, 1.0, 0.0]
