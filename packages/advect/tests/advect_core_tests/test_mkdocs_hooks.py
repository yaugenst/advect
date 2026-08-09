"""Test generated-documentation validation helpers."""

from __future__ import annotations

from scripts.mkdocs_hooks import _api_rendering_errors, _malformed_markdown_table_lines


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
