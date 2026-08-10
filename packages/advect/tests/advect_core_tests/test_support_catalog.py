"""Tests for the runtime-derived extension support catalog."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.report_extension_support import (
    _render_frontend_table,
    render_pages,
)

import advect as ad
from advect.core._array_api.frontend import (
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
)
from advect.core._registry import get_registry
from advect.numpy._support_contract import numpy_support_declarations
from advect.support import (
    _scipy_function_row,
    _scipy_primitive_lowering,
    _walk_public_functions,
)

_REMOVED_CONTRACT_FIELDS = frozenset(
    {"complete", "modes", "note", "parameters", "signature", "signature_status"}
)


def _extensions() -> dict[str, dict[str, object]]:
    return ad.support_catalog()["extensions"]


def _rows(extension: str) -> list[dict[str, object]]:
    return list(_extensions()[extension]["functions"])


def _by_callable(extension: str) -> dict[str, dict[str, object]]:
    return {str(row["callable"]): row for row in _rows(extension)}


def test_catalog_is_json_serializable_and_contains_no_signature_contract_metadata() -> None:
    catalog = ad.support_catalog()

    assert catalog["schema_version"] == 3
    assert set(catalog["extensions"]) == {"array_api", "numpy", "scipy"}
    assert json.loads(json.dumps(catalog)) == catalog
    for extension in catalog["extensions"].values():
        for row in extension["functions"]:
            assert not _REMOVED_CONTRACT_FIELDS & row.keys()
            assert {"dynamic", "staged", "serialized"} <= row.keys()


def test_primitive_matrix_is_a_live_registry_projection() -> None:
    catalog = ad.support_catalog()
    rows = {str(row["primitive"]): row for row in catalog["primitives"]}
    definitions = {
        definition.name
        for definition in get_registry().definitions()
        if definition.name.startswith(("advect.", "array.", "array_ext.", "custom.scipy."))
    }

    assert rows.keys() == definitions
    assert rows["array.sin"]["jvp"] == "yes"
    assert rows["array.sin"]["abstract"] is True
    assert rows["array.sum"]["evaluator"] is True
    assert rows["custom.scipy.special.erf"]["vjp"] in {"direct", "from JVP"}


def test_array_api_catalog_projects_the_live_binding_table() -> None:
    rows = _by_callable("array_api")

    assert rows.keys() == (
        set(_FUNCTION_SPECS) | _ARRAY_API_COMPOSITES | set(_ARRAY_API_META_FUNCTIONS)
    )
    assert rows["sin"]["lowering"] == "array.sin"
    assert rows["sin"]["backed_by"] == "array_api"
    assert rows["sin"]["staged"] is True
    assert rows["sin"]["serialized"] is True
    assert rows["finfo"]["lowering"] == "metadata"
    assert rows["finfo"]["jvp"] == "n/a"
    assert rows["meshgrid"]["lowering"] == "composite"
    assert rows["meshgrid"]["staged"] is True
    assert rows["unique_all"]["dynamic"] is True
    assert rows["unique_all"]["staged"] is False
    assert rows["unique_all"]["abstract"] == "no"
    assert rows["nonzero"]["jvp"] == "n/a"


def test_numpy_catalog_is_bounded_by_executable_evidence() -> None:
    rows = _rows("numpy")
    declared = {
        (declaration.kind, declaration.callable) for declaration in numpy_support_declarations()
    }

    assert {(str(row["kind"]), str(row["callable"])) for row in rows} <= declared
    assert all(row["dynamic"] is True for row in rows)
    assert "numpy.arange" in {row["callable"] for row in rows}
    assert not any(str(row["callable"]).endswith((".reduceat", ".at")) for row in rows)


def test_numpy_catalog_keeps_value_dependent_scimath_dynamic_only() -> None:
    rows = _by_callable("numpy")
    scimath_rows = [row for path, row in rows.items() if path.startswith("numpy.lib.scimath.")]

    assert len(scimath_rows) == 9
    assert all(row["dynamic"] is True for row in scimath_rows)
    assert all(row["staged"] is False for row in scimath_rows)
    assert all(row["serialized"] is False for row in scimath_rows)
    assert "numpy.polyval" in rows
    assert "numpy.polynomial.polynomial.polyval" not in rows

    page = render_pages(ad.support_catalog())["numpy.md"]
    assert "The nine `numpy.lib.scimath` rows are dynamic-only" in page
    assert "`numpy.round` supports staging and serialization" in page
    assert "`numpy.linalg.eig` and `numpy.linalg.eigvals` support all three" in page


def test_numpy_catalog_marks_array_api_reuse_without_signature_inheritance() -> None:
    rows = _by_callable("numpy")

    assert rows["numpy.sin"]["lowering"] == "array.sin"
    assert rows["numpy.sin"]["backed_by"] == "array_api"
    assert rows["numpy.linalg.diagonal"]["backed_by"] == "array_api"
    assert rows["numpy.linalg.matrix_power"]["lowering"] == "composite"
    assert rows["numpy.linalg.matrix_power"]["jvp"] == "composite"
    assert rows["numpy.ndarray.copy"]["lowering"] == "advect.copy"
    assert rows["numpy.ndarray.copy"]["backed_by"] == "advect"
    assert "signature_status" not in rows["numpy.ndarray.copy"]


def test_scipy_catalog_is_discovered_from_public_extension_exports() -> None:
    import advect.scipy as scipy_extension  # noqa: PLC0415

    rows = _rows("scipy")
    functions = {str(row["entrypoint"]) for row in rows if row["kind"] == "function"}
    adapters = {str(row["entrypoint"]) for row in rows if row["kind"] == "adapter"}
    public = _walk_public_functions(scipy_extension)
    expected_functions = set()
    expected_adapters = set()
    for function in public:
        entrypoint = f"{function.__module__}.{function.__name__}"
        row = _scipy_function_row(function)
        target = expected_functions if row["kind"] == "function" else expected_adapters
        target.add(entrypoint)

    assert functions == expected_functions
    assert adapters == expected_adapters
    public_by_entrypoint = {
        f"{function.__module__}.{function.__name__}": function for function in public
    }
    for entrypoint in expected_functions:
        callable_path = entrypoint.replace("advect.scipy", "scipy", 1)
        function = public_by_entrypoint[entrypoint]
        expected_lowering = (
            "composite"
            if getattr(function, "__advect_lowering__", None) == "composite"
            else _scipy_primitive_lowering(entrypoint)
        )
        assert _by_callable("scipy")[callable_path]["lowering"] == expected_lowering
    assert all(row["staged"] is True for row in rows if row["kind"] == "function")
    assert all(row["serialized"] is True for row in rows if row["kind"] == "function")
    assert all(row["staged"] is False for row in rows if row["kind"] == "adapter")
    assert all(row["serialized"] is False for row in rows if row["kind"] == "adapter")


def test_scipy_catalog_recognizes_public_composite_markers() -> None:
    def opening(value: object) -> object:
        return value

    opening.__module__ = "advect.scipy.ndimage"
    opening.__advect_lowering__ = "composite"  # type: ignore[attr-defined]

    row = _scipy_function_row(opening)

    assert row == {
        "abstract": "composite",
        "backed_by": "composite",
        "callable": "scipy.ndimage.opening",
        "dynamic": True,
        "entrypoint": "advect.scipy.ndimage.opening",
        "jvp": "composite",
        "kind": "function",
        "lowering": "composite",
        "serialized": True,
        "staged": True,
        "vjp": "composite",
    }


def test_checked_in_compatibility_pages_are_generated_from_the_live_catalog() -> None:
    repository = Path(__file__).resolve().parents[4]

    for name, content in render_pages(ad.support_catalog()).items():
        document = repository / "docs" / "compatibility" / name
        assert document.read_text(encoding="utf-8") == content, name


def test_compatibility_tables_show_user_capabilities() -> None:
    pages = render_pages(ad.support_catalog())

    for name in ("numpy.md", "array-api.md", "scipy.md"):
        page = pages[name]
        assert "| Function | Stage/save | Differentiate |" in page
        assert "Lowers to" not in page
        assert "compat-columns" not in page

    array_api = pages["array-api.md"]
    assert "**No** means no derivative rule is available" in array_api
    assert "| `add` | yes | yes |" in array_api
    assert "| `all` | yes | n/a |" in array_api
    assert "| `arange` | yes | no |" in array_api


def test_compact_table_preserves_asymmetric_capabilities() -> None:
    rows = [
        {
            "callable": "forward",
            "dynamic": True,
            "staged": True,
            "serialized": False,
            "jvp": "yes",
            "vjp": "no",
        },
        {
            "callable": "reverse",
            "dynamic": True,
            "staged": False,
            "serialized": True,
            "jvp": "no",
            "vjp": "direct",
        },
    ]

    table = "\n".join(_render_frontend_table(rows))

    assert "| `forward` | stage only | forward only |" in table
    assert "| `reverse` | save only | reverse only |" in table
    with pytest.raises(ValueError, match="every row to support dynamic"):
        _render_frontend_table([{**rows[0], "dynamic": False}])


def test_cupy_page_is_honest_without_duplicating_the_array_api_catalog() -> None:
    page = render_pages(ad.support_catalog())["cupy.md"]

    assert "not yet a release claim" in page
    assert "## Functions" not in page
    assert "| Function | Stage/save | Differentiate |" not in page


def test_numpy_version_is_live() -> None:
    assert _extensions()["numpy"]["version"] == np.__version__
