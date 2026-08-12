"""Additional Array API evidence and frontend boundary contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core import _backends
from advect.core._array_api import support
from advect.core._array_api.evidence import (
    ArrayInput,
    Input,
    MetadataCase,
    OperationCase,
    case_parameter_values,
    input_indices,
    operation_evidence_cases,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    minimum_array_api_version,
)


def _trace(function: Any, value: Any) -> Any:
    traced = trace_call(
        function,
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        return traced.output
    finally:
        traced.tape.release_payloads()


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((Input(0), Input(0)), (), "too many positional arguments"),
        ((Input(0),), (("x", Input(0)),), "binds 'x' more than once"),
        ((), (), "does not bind required parameter 'x'"),
        ((Input(0),), (("unknown", None),), "binds unknown parameters"),
    ],
)
def test_evidence_cases_reject_calls_outside_the_official_signature(
    args: tuple[object, ...],
    kwargs: tuple[tuple[str, object], ...],
    match: str,
) -> None:
    case = OperationCase(
        "abs",
        (ArrayInput([1.0], "float64"),),
        args,
        kwargs,
        variant="invalid",
    )

    with pytest.raises(ValueError, match=match):
        case_parameter_values(case)


def test_evidence_input_references_are_found_inside_nested_values() -> None:
    value = {"left": [Input(2), (Input(0),)], "configuration": {"axis": 0}}

    assert input_indices(value) == frozenset({0, 2})


def test_unknown_callable_has_no_minimum_supported_revision() -> None:
    with pytest.raises(KeyError, match="future_operation"):
        minimum_array_api_version("future_operation")


def test_support_profile_fails_closed_for_missing_metadata_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = support.metadata_cases()
    monkeypatch.setattr(
        support,
        "metadata_cases",
        lambda: tuple(case for case in cases if case.path != "finfo"),
    )

    row = next(
        row for row in support.build_support_profile()["callables"] if row["path"] == "finfo"
    )

    assert row["complete"] is False
    assert row["modes"] == []
    assert row["note"] == "no executable metadata evidence"


def test_support_profile_fails_closed_for_incomplete_metadata_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = support.metadata_cases()
    monkeypatch.setattr(
        support,
        "metadata_cases",
        lambda: tuple(
            MetadataCase(case.path, case.data, case.dtype, (), ("dynamic",))
            if case.path == "finfo"
            else case
            for case in cases
        ),
    )

    row = next(
        row for row in support.build_support_profile()["callables"] if row["path"] == "finfo"
    )

    assert row["complete"] is False
    assert "metadata parameters lack executable evidence" in row["note"]
    assert "metadata lifetime evidence is incomplete" in row["note"]


def test_support_profile_requires_baseline_and_live_parameter_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = operation_evidence_cases(
        support._static_parameters(version=LATEST_ARRAY_API_VERSION),
        LATEST_ARRAY_API_VERSION,
    )
    weakened = tuple(
        replace(case, args=(object(),), variant="constant-input") if case.path == "abs" else case
        for case in cases
    )
    monkeypatch.setattr(
        support,
        "operation_evidence_cases",
        lambda _static_parameters, _version: weakened,
    )

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == "abs")

    assert row["complete"] is False
    assert "no baseline callable evidence" in row["note"]
    assert "x lacks live-parameter evidence" in row["note"]


@pytest.mark.parametrize(
    ("schema", "path"),
    [
        (None, "abs"),
        (SimpleNamespace(allowed_attrs=frozenset(), positional_attrs=frozenset()), "sum"),
    ],
    ids=("missing-abstract-schema", "missing-static-attribute"),
)
def test_support_profile_requires_complete_abstract_lowering(
    monkeypatch: pytest.MonkeyPatch,
    schema: object,
    path: str,
) -> None:
    registry = SimpleNamespace(
        get_optional=lambda _name: SimpleNamespace(abstract_schema=schema),
    )
    monkeypatch.setattr(support, "get_registry", lambda: registry)

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == path)

    assert row["complete"] is False
    assert row["modes"] == []
    assert "claimed lifetimes lack executable evidence" in row["note"]


def test_cumulative_initial_uses_the_vector_default_axis() -> None:
    value = strict.asarray([1.0, 2.0, 3.0], dtype=strict.float64)

    actual = _trace(
        lambda x: x.__array_namespace__().cumulative_sum(x, include_initial=True),
        value,
    )

    np.testing.assert_array_equal(np.asarray(actual), [0.0, 1.0, 3.0, 6.0])


def test_live_sequence_accepts_an_empty_array_child() -> None:
    value = strict.asarray([], dtype=strict.float64)

    actual = _trace(
        lambda x: x.__array_namespace__().asarray([x, []], dtype=x.dtype),
        value,
    )

    assert actual.shape == (2, 0)
    assert actual.dtype == strict.float64


def test_debug_representation_summarizes_the_traced_provider_value() -> None:
    value = strict.asarray([1.0, 2.0], dtype=strict.float64)
    representations: list[str] = []

    with ad.debug():
        _trace(lambda x: representations.append(repr(x)) or x, value)

    assert len(representations) == 1
    assert "ArrayAPITracer(node=%" in representations[0]
    assert "float64" in representations[0]


def test_foreign_array_protocols_defer_when_no_frontend_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = strict.asarray([1.0, 2.0], dtype=strict.float64)
    observed: list[object] = []
    original_get_hook = _backends.get_hook
    monkeypatch.setattr(
        _backends,
        "get_hook",
        lambda name: None if name.startswith("advect.foreign_array_") else original_get_hook(name),
    )

    def inspect(x: Any) -> Any:
        observed.extend(
            (
                x.__array_ufunc__(object(), "__call__", x),
                x.__array_function__(object(), (type(x),), (x,), {}),
            )
        )
        return x

    _trace(inspect, value)

    assert observed == [NotImplemented, NotImplemented]
