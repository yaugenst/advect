"""Additional Array API evidence and frontend boundary contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import advect as ad
from advect.core._array_api import support
from advect.core._array_api.evidence import (
    MetadataCase,
    operation_evidence_cases,
)
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable


def test_array_api_catalog_modes_are_the_evidenced_support_profile() -> None:
    # Regression: the catalog claimed staged/serialized whenever an abstract
    # rule existed, e.g. for default-dtype `zeros`, which cannot stage.
    catalog = ad.support_catalog()["extensions"]["array_api"]["functions"]
    profile = {str(row["path"]): row for row in support.build_support_profile()["callables"]}
    evidence: dict[str, list[tuple[str, ...]]] = {}
    for case in (
        *operation_evidence_cases(support._static_parameters(version=LATEST_ARRAY_API_VERSION)),
        *support.metadata_cases(),
    ):
        evidence.setdefault(case.path, []).append(case.modes)
    partial_notes = set(support._PARTIAL_PARAMETERS.values())

    for row in catalog:
        path = str(row["callable"])
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
