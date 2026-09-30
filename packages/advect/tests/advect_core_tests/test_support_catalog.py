"""Tests for the runtime-derived extension support catalog and its evidence profile."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scripts.report_extension_support import (
    _render_frontend_table,
    render_pages,
)

import advect as ad
from advect.core._array_api import support
from advect.core._array_api.evidence import MetadataCase, operation_evidence_cases
from advect.core._array_api.frontend import (
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
)
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.core._registry import get_registry
from advect.support import _scipy_function_row, _walk_public_functions

if TYPE_CHECKING:
    from collections.abc import Callable

_REMOVED_CONTRACT_FIELDS = frozenset(
    {"complete", "modes", "note", "parameters", "signature", "signature_status"}
)


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    """Build the catalog once; these tests only read it."""
    return ad.support_catalog()


@pytest.fixture(scope="module")
def pages(catalog: dict[str, Any]) -> dict[str, str]:
    return render_pages(catalog)


def _by_callable(catalog: dict[str, Any], extension: str) -> dict[str, dict[str, object]]:
    return {str(row["callable"]): row for row in catalog["extensions"][extension]["functions"]}


def test_catalog_is_json_serializable_and_contains_no_signature_contract_metadata(
    catalog: dict[str, Any],
) -> None:
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


def test_array_api_catalog_projects_the_live_binding_table(catalog: dict[str, Any]) -> None:
    rows = _by_callable(catalog, "array_api")

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


def test_array_api_catalog_modes_are_the_evidenced_support_profile(
    catalog: dict[str, Any],
) -> None:
    # Regression: the catalog claimed staged/serialized whenever an abstract
    # rule existed, e.g. for default-dtype `zeros`, which cannot stage.
    profile = {str(row["path"]): row for row in support.build_support_profile()["callables"]}
    evidence: dict[str, list[tuple[str, ...]]] = {}
    for case in (
        *operation_evidence_cases(support._static_parameters(version=LATEST_ARRAY_API_VERSION)),
        *support.metadata_cases(),
    ):
        evidence.setdefault(case.path, []).append(case.modes)
    partial_notes = set(support._PARTIAL_PARAMETERS.values())

    for path, row in _by_callable(catalog, "array_api").items():
        claimed = [mode for mode in ("dynamic", "staged", "serialized") if row[mode]]
        declared = profile[path]
        assert all(set(claimed) <= set(modes) for modes in evidence[path]), path
        if declared["complete"]:
            assert claimed == declared["modes"], path
        else:
            assert set(str(declared["note"]).split("; ")) <= partial_notes, path


def _evidence(transform: Callable[[Any], Any]) -> Callable[[pytest.MonkeyPatch], None]:
    """Patch callable evidence; ``transform`` returns ``None`` to drop a case."""

    def patch(monkeypatch: pytest.MonkeyPatch) -> None:
        cases = operation_evidence_cases(
            support._static_parameters(version=LATEST_ARRAY_API_VERSION),
            LATEST_ARRAY_API_VERSION,
        )
        kept = tuple(case for case in map(transform, cases) if case is not None)
        monkeypatch.setattr(support, "operation_evidence_cases", lambda _static, _version: kept)

    return patch


def _metadata(transform: Callable[[Any], Any]) -> Callable[[pytest.MonkeyPatch], None]:
    """Patch metadata evidence; ``transform`` returns ``None`` to drop a case."""

    def patch(monkeypatch: pytest.MonkeyPatch) -> None:
        kept = tuple(case for case in map(transform, support.metadata_cases()) if case is not None)
        monkeypatch.setattr(support, "metadata_cases", lambda: kept)

    return patch


def _abstract_schema(schema: object) -> Callable[[pytest.MonkeyPatch], None]:
    """Give every canonical operation the same abstract lowering schema."""

    def patch(monkeypatch: pytest.MonkeyPatch) -> None:
        registry = SimpleNamespace(
            get_optional=lambda _name: SimpleNamespace(abstract_schema=schema)
        )
        monkeypatch.setattr(support, "get_registry", lambda: registry)

    return patch


def _replaced(path_or_identifier: str, replacement: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda case: (
        replacement(case) if path_or_identifier in {case.path, case.identifier} else case
    )


_KEEPDIMS_DEFAULT = "sum[keepdims=default]"


@pytest.mark.parametrize(
    ("patch", "path", "notes"),
    [
        pytest.param(
            _metadata(_replaced("finfo", lambda _case: None)),
            "finfo",
            {"no executable metadata evidence"},
            id="metadata-evidence-absent",
        ),
        pytest.param(
            _metadata(
                _replaced(
                    "finfo",
                    lambda case: MetadataCase(case.path, case.data, case.dtype, (), ("dynamic",)),
                )
            ),
            "finfo",
            {
                "metadata lifetime evidence is incomplete",
                "metadata parameters lack executable evidence",
            },
            id="metadata-evidence-incomplete",
        ),
        pytest.param(
            _evidence(_replaced("abs", lambda _case: None)),
            "abs",
            {"no executable callable evidence"},
            id="callable-evidence-absent",
        ),
        pytest.param(
            _evidence(
                _replaced(
                    "abs",
                    lambda case: replace(case, args=(object(),), variant="constant-input"),
                )
            ),
            "abs",
            {"no baseline callable evidence", "x lacks live-parameter evidence"},
            id="baseline-and-live-parameter-absent",
        ),
        pytest.param(
            _evidence(_replaced(_KEEPDIMS_DEFAULT, lambda _case: None)),
            "sum",
            {"keepdims lacks default static-variant evidence"},
            id="static-variant-absent",
        ),
        pytest.param(
            _evidence(_replaced(_KEEPDIMS_DEFAULT, lambda case: replace(case, modes=("dynamic",)))),
            "sum",
            {"claimed lifetimes lack executable evidence"},
            id="variant-lifetime-absent",
        ),
        pytest.param(
            _abstract_schema(None),
            "abs",
            {"claimed lifetimes lack executable evidence"},
            id="abstract-schema-absent",
        ),
        pytest.param(
            _abstract_schema(
                SimpleNamespace(allowed_attrs=frozenset(), positional_attrs=frozenset())
            ),
            "sum",
            {"claimed lifetimes lack executable evidence"},
            id="static-attribute-unlowered",
        ),
    ],
)
def test_support_profile_fails_closed_without_complete_evidence(
    monkeypatch: pytest.MonkeyPatch,
    patch: Callable[[pytest.MonkeyPatch], None],
    path: str,
    notes: set[str],
) -> None:
    patch(monkeypatch)

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == path)

    assert row["complete"] is False
    assert row["modes"] == []
    assert set(str(row["note"]).split("; ")) == notes


def test_numpy_catalog_keeps_value_dependent_scimath_dynamic_only(
    catalog: dict[str, Any],
    pages: dict[str, str],
) -> None:
    rows = _by_callable(catalog, "numpy")
    scimath_rows = [row for path, row in rows.items() if path.startswith("numpy.lib.scimath.")]

    assert len(scimath_rows) == 9
    assert all(row["dynamic"] is True for row in scimath_rows)
    assert all(row["staged"] is False for row in scimath_rows)
    assert all(row["serialized"] is False for row in scimath_rows)
    assert "numpy.polyval" in rows
    assert "numpy.polynomial.polynomial.polyval" not in rows

    page = pages["numpy.md"]
    assert "All `numpy.lib.scimath` rows are dynamic-only" in page
    assert "`numpy.round` supports staging and serialization" in page
    assert "`numpy.linalg.eig` and `numpy.linalg.eigvals` support all three" in page


def test_numpy_catalog_marks_array_api_reuse_without_signature_inheritance(
    catalog: dict[str, Any],
) -> None:
    rows = _by_callable(catalog, "numpy")

    assert rows["numpy.sin"]["lowering"] == "array.sin"
    assert rows["numpy.sin"]["backed_by"] == "array_api"
    assert rows["numpy.linalg.diagonal"]["backed_by"] == "array_api"
    assert rows["numpy.linalg.matrix_power"]["lowering"] == "composite"
    assert rows["numpy.linalg.matrix_power"]["jvp"] == "composite"
    assert rows["numpy.ndarray.copy"]["lowering"] == "advect.copy"
    assert rows["numpy.ndarray.copy"]["backed_by"] == "advect"
    assert "signature_status" not in rows["numpy.ndarray.copy"]


def test_scipy_catalog_is_discovered_from_public_extension_exports(
    catalog: dict[str, Any],
) -> None:
    import advect.scipy as scipy_extension  # noqa: PLC0415

    public = {
        f"{function.__module__}.{function.__name__}": function
        for function in _walk_public_functions(scipy_extension)
    }
    rows = {str(row["entrypoint"]): row for row in catalog["extensions"]["scipy"]["functions"]}
    registry = get_registry()

    assert rows.keys() == public.keys()
    for entrypoint, row in rows.items():
        primitive = f"custom.scipy.{entrypoint.removeprefix('advect.scipy.')}"
        if registry.has(primitive):
            lowering = primitive
        elif getattr(public[entrypoint], "__advect_lowering__", None) == "composite":
            lowering = "composite"
        else:
            lowering = "adapter"
        function = lowering != "adapter"
        assert row["lowering"] == lowering, entrypoint
        assert row["kind"] == ("function" if function else "adapter"), entrypoint
        assert row["callable"] == (
            entrypoint.replace("advect.scipy", "scipy", 1) if function else entrypoint
        )
        assert row["staged"] is function, entrypoint
        assert row["serialized"] is function, entrypoint


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


def test_checked_in_compatibility_pages_are_generated_from_the_live_catalog(
    pages: dict[str, str],
) -> None:
    repository = Path(__file__).resolve().parents[4]

    for name, content in pages.items():
        document = repository / "docs" / "compatibility" / name
        assert document.read_text(encoding="utf-8") == content, name


def test_compatibility_tables_show_user_capabilities(pages: dict[str, str]) -> None:
    for name in ("numpy.md", "array-api.md", "scipy.md"):
        page = pages[name]
        assert "| Function | Stage/save | Differentiate |" in page
        assert "Lowers to" not in page
        assert "compat-columns" not in page

    array_api = pages["array-api.md"]
    assert "**No** means no derivative rule is available" in array_api
    assert "| `add` | yes | yes |" in array_api
    assert "| `all` | yes | n/a |" in array_api
    assert "| `arange` | no | no |" in array_api


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


def test_cupy_page_is_honest_without_duplicating_the_array_api_catalog(
    pages: dict[str, str],
) -> None:
    page = pages["cupy.md"]

    assert "does not maintain a second CuPy inventory" in page
    assert "## Functions" not in page
    assert "| Function | Stage/save | Differentiate |" not in page


def test_numpy_version_is_live(catalog: dict[str, Any]) -> None:
    assert catalog["extensions"]["numpy"]["version"] == np.__version__
