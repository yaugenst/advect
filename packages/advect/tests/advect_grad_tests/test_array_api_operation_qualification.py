"""Execution qualification for the declared staged Array API surface."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import array_api_strict
import numpy as np
import pytest
from scripts import qualify_array_api_operations as qualifier

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api import support
from advect.core._array_api.evidence import (
    case_parameter_values,
    input_indices,
    metadata_cases,
    operation_evidence_cases,
)
from advect.core._array_api.frontend import (
    _ARRAY_API_META_FUNCTIONS,
    _DYNAMIC_ARRAY_API_COMPOSITES,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._array_api.support import (
    _evidence_gaps,
    _static_parameters,
    build_support_profile,
)

_EVIDENCE = {
    version: operation_evidence_cases(_static_parameters(version=version), version)
    for version in SUPPORTED_ARRAY_API_VERSIONS
}
_ALL_CASES = _EVIDENCE[LATEST_ARRAY_API_VERSION]
_PORTABLE_CASES = tuple(case for case in _ALL_CASES if case.portable)
_BASELINE_CASES = {case.path: case for case in _ALL_CASES if case.variant == "baseline"}
_ROWS = {str(row["path"]): row for row in build_support_profile()["callables"]}
_REVISION_CASES = tuple(
    pytest.param(version, case, id=f"{version}-{case.identifier}")
    for version, cases in _EVIDENCE.items()
    for case in cases
)


def _derivative_cases() -> tuple[Any, ...]:
    """Pair each complete baseline with each differentiable parameter per revision."""
    cases = []
    for version, evidence in _EVIDENCE.items():
        rows = {str(row["path"]): row for row in build_support_profile(version)["callables"]}
        for case in evidence:
            if case.variant != "baseline" or rows[case.path]["complete"] is not True:
                continue
            values = case_parameter_values(case, version)
            cases.extend(
                pytest.param(
                    version,
                    case,
                    name,
                    tuple(sorted(input_indices(values[name]))),
                    id=f"{version}-{case.path}-{name}",
                )
                for parameter in rows[case.path]["parameters"]
                if parameter["role"] == "differentiable"
                for name in (str(parameter["name"]),)
            )
    return tuple(cases)


_DERIVATIVE_CASES = _derivative_cases()


def test_case_catalog_covers_every_claimed_non_metadata_lifetime() -> None:
    complete_rows = {
        str(row["path"]): row
        for row in _ROWS.values()
        if row["complete"] is True and row["path"] not in _ARRAY_API_META_FUNCTIONS
    }
    case_paths = set(_BASELINE_CASES)

    assert len({case.identifier for case in _ALL_CASES}) == len(_ALL_CASES)
    assert set(complete_rows) - case_paths == set()
    assert case_paths - set(complete_rows) == {"repeat"}
    for path, row in complete_rows.items():
        path_cases = [case for case in _ALL_CASES if case.path == path]
        assert path_cases
        assert all(tuple(row["modes"]) == case.modes for case in path_cases), path


def test_complete_claims_have_no_evidence_gaps() -> None:
    gaps = _evidence_gaps(version=LATEST_ARRAY_API_VERSION)

    assert all(not gaps[path] for path, row in _ROWS.items() if row["complete"] is True)
    assert all(
        gaps[path] or "unsupported" in {parameter["role"] for parameter in row["parameters"]}
        for path, row in _ROWS.items()
        if row["complete"] is False
    )


def test_complete_claim_fails_closed_when_callable_evidence_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remaining = tuple(case for case in _ALL_CASES if case.path != "abs")
    monkeypatch.setattr(
        support,
        "operation_evidence_cases",
        lambda _static_parameters, _version: remaining,
    )

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == "abs")

    assert row["complete"] is False
    assert row["modes"] == []
    assert row["note"] == "no executable callable evidence"


def test_complete_claim_fails_closed_when_static_variant_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remaining = tuple(
        case
        for case in _ALL_CASES
        if not (case.path == "sum" and case.variant == "keepdims=default")
    )
    monkeypatch.setattr(
        support,
        "operation_evidence_cases",
        lambda _static_parameters, _version: remaining,
    )

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == "sum")

    assert row["complete"] is False
    assert row["modes"] == []
    assert "keepdims lacks default static-variant evidence" in row["note"]


def test_complete_claim_fails_closed_when_one_variant_lacks_a_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weakened = tuple(
        replace(case, modes=("dynamic",))
        if case.path == "sum" and case.variant == "keepdims=default"
        else case
        for case in _ALL_CASES
    )
    monkeypatch.setattr(
        support,
        "operation_evidence_cases",
        lambda _static_parameters, _version: weakened,
    )

    row = next(row for row in support.build_support_profile()["callables"] if row["path"] == "sum")

    assert row["complete"] is False
    assert row["modes"] == []
    assert "claimed lifetimes lack executable evidence" in row["note"]


def _metadata_can_cast(x: Any) -> object:
    namespace = x.__array_namespace__()
    dtype = namespace.float64 if namespace.can_cast(x.dtype, namespace.float64) else x.dtype
    return namespace.astype(x, dtype)


def _metadata_finfo(x: Any) -> object:
    namespace = x.__array_namespace__()
    epsilon = namespace.asarray(namespace.finfo(x.dtype).eps, dtype=x.dtype)
    return namespace.add(x, epsilon)


def _metadata_iinfo(x: Any) -> object:
    namespace = x.__array_namespace__()
    bits = namespace.asarray(namespace.iinfo(x.dtype).bits, dtype=x.dtype)
    return namespace.add(x, bits)


def _metadata_isdtype(x: Any) -> object:
    namespace = x.__array_namespace__()
    if namespace.isdtype(x.dtype, "real floating"):
        return namespace.negative(x)
    return namespace.positive(x)


def _metadata_result_type(x: Any) -> object:
    namespace = x.__array_namespace__()
    return namespace.astype(x, namespace.result_type(x.dtype, namespace.float64))


@pytest.mark.parametrize(
    "case",
    metadata_cases(),
    ids=lambda case: case.identifier,
)
def test_compile_time_metadata_controls_dynamic_trace_and_serialized_stage(
    case: Any,
) -> None:
    functions = {
        "can_cast": _metadata_can_cast,
        "finfo": _metadata_finfo,
        "iinfo": _metadata_iinfo,
        "isdtype": _metadata_isdtype,
        "result_type": _metadata_result_type,
    }
    function = functions[case.path]
    namespace = array_api_strict
    value = namespace.asarray(case.data, dtype=getattr(namespace, case.dtype))
    expected = function(value)

    trace = trace_call(
        function,
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        dynamic = trace.output
    finally:
        trace.tape.release_payloads()
    program = ad.stage(
        function,
        specs=(ad.ArraySpec(tuple(value.shape), value.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    outputs = {
        "dynamic": dynamic,
        "serialized": restored(value),
        "staged": program(value),
    }
    assert set(outputs) == set(case.modes)
    for actual in outputs.values():
        qualifier.assert_output_matches(expected, actual, compare_values=True)
    assert case.path in _ARRAY_API_META_FUNCTIONS


def test_generated_report_records_common_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _EVIDENCE["2022.12"][0]
    monkeypatch.setattr(
        qualifier,
        "operation_evidence_cases",
        lambda _static_parameters, _version: (case,),
    )

    report = qualifier.build_report(
        provider_names=("array-api-strict",),
        subset="all",
        array_api_version="2022.12",
    )

    assert report["schema_version"] == 1
    assert report["report_kind"] == "advect.array-api-operation-qualification"
    assert report["api_version"] == "2022.12"
    assert report["passed"] is True
    assert report["providers"][0]["reported_array_api_version"] == "2022.12"
    assert report["providers"][0]["cases"][0]["lifetimes"] == list(case.modes)
    assert report["environment"]["source_revision"]
    assert report["environment"]["python"]
    assert report["environment"]["machine"]["platform"]


def test_revision_restriction_restores_the_reference_provider_flags() -> None:
    # Regression: selecting a revision left array-api-strict at that revision,
    # breaking later tests that call functions added after it.
    before = array_api_strict.get_array_api_strict_flags()
    provider = qualifier._provider("array-api-strict", "2022.12")

    assert provider.reported_array_api_version == "2022.12"
    assert array_api_strict.get_array_api_strict_flags() == before
    with qualifier._restrict_provider_revision(provider, "2022.12"):
        assert array_api_strict.__array_api_version__ == "2022.12"
    assert array_api_strict.get_array_api_strict_flags() == before


@pytest.mark.parametrize(("array_api_version", "case"), _REVISION_CASES)
def test_declared_case_round_trips_on_array_api_strict(
    array_api_version: str,
    case: Any,
) -> None:
    provider = qualifier._provider("array-api-strict", array_api_version)
    expected, outputs = qualifier.execute_lifetimes(
        case,
        provider,
        array_api_version=array_api_version,
    )

    assert set(outputs) == set(case.modes)
    for output in outputs.values():
        qualifier.assert_output_matches(
            expected,
            output,
            compare_values=case.compare_values,
        )


@pytest.mark.parametrize("path", sorted(_DYNAMIC_ARRAY_API_COMPOSITES))
def test_data_dependent_composites_reject_abstract_staging(path: str) -> None:
    case = _BASELINE_CASES[path]
    provider = qualifier._provider("array-api-strict")
    inputs = qualifier._materialize_inputs(case, provider.namespace)
    function = qualifier._transformed(case)

    with pytest.raises(NotImplementedError, match="no abstract staging rule"):
        ad.stage(
            function,
            specs=tuple(qualifier._array_spec(value) for value in inputs),
        )


@pytest.mark.parametrize("case", _PORTABLE_CASES, ids=lambda case: case.path)
def test_portable_case_round_trips_on_numpy(case: object) -> None:
    provider = qualifier._provider("numpy")
    expected, outputs = qualifier.execute_lifetimes(case, provider)

    for output in outputs.values():
        qualifier.assert_output_matches(
            expected,
            output,
            compare_values=case.compare_values,
        )


def _ones_like_tree(value: object, namespace: object) -> object:
    if isinstance(value, tuple):
        items = tuple(_ones_like_tree(item, namespace) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items
    if isinstance(value, list):
        return [_ones_like_tree(item, namespace) for item in value]
    return namespace.ones_like(value)


def _tree_leaves(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple | list):
        return tuple(leaf for item in value for leaf in _tree_leaves(item))
    return (value,)


def _scaled_direction(value: object, scale: float, namespace: object) -> object:
    scalar = namespace.asarray(scale, dtype=value.dtype)
    return namespace.multiply(namespace.ones_like(value), scalar)


def _real_pairing(left: object, right: object) -> float:
    return float(
        sum(
            np.real(np.vdot(np.asarray(left_leaf), np.asarray(right_leaf)))
            for left_leaf, right_leaf in zip(
                _tree_leaves(left),
                _tree_leaves(right),
                strict=True,
            )
        )
    )


@pytest.mark.parametrize(
    ("array_api_version", "case", "parameter", "argnums"),
    _DERIVATIVE_CASES,
)
@pytest.mark.parametrize("scale", [-0.75, 0.5, 1.5])
def test_each_differentiable_parameter_executes_jvp_and_vjp(
    array_api_version: str,
    case: Any,
    parameter: str,
    argnums: tuple[int, ...],
    scale: float,
) -> None:
    path = case.path
    provider = qualifier._provider("array-api-strict", array_api_version)
    with qualifier._restrict_provider_revision(provider, array_api_version):
        inputs = qualifier._materialize_inputs(case, provider.namespace)
        function = qualifier._transformed(case)
        expected = qualifier._invoke(case, provider.namespace, inputs)
        tangents = tuple(
            _scaled_direction(inputs[index], scale, provider.namespace) for index in argnums
        )

        primal, directional = ad.jvp(function, argnums=argnums)(
            *inputs,
            tangents=tangents,
        )
        qualifier.assert_output_matches(expected, primal, compare_values=case.compare_values)

        value, pullback = ad.vjp(function, argnums=argnums)(*inputs)
        seed = _ones_like_tree(value, provider.namespace)
        try:
            cotangents = pullback(seed)
        finally:
            pullback.close()
        cotangent_items = cotangents if isinstance(cotangents, tuple) else (cotangents,)
        assert len(cotangent_items) == len(argnums), parameter
        for cotangent, index in zip(cotangent_items, argnums, strict=True):
            assert tuple(cotangent.shape) == tuple(inputs[index].shape), parameter
            assert cotangent.dtype == inputs[index].dtype, parameter
        reverse = cotangent_items if len(argnums) > 1 else cotangent_items[0]
        forward_pairing = _real_pairing(seed, directional)
        reverse_pairing = _real_pairing(
            reverse,
            tangents if len(argnums) > 1 else tangents[0],
        )
        np.testing.assert_allclose(
            reverse_pairing,
            forward_pairing,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{path}.{parameter} violates the JVP/VJP adjoint identity",
        )

        if "serialized" not in case.modes:
            return
        primal_program = ad.stage(
            function,
            specs=tuple(qualifier._array_spec(value) for value in inputs),
            array_api_version=array_api_version,
        )
        derivative_program = ad.vjp_program(primal_program, argnums=argnums)
        restored_derivative = ad.StagedProgram.from_dict(derivative_program.to_dict())

        assert derivative_program.array_api_version == array_api_version
        for program in (derivative_program, restored_derivative):
            staged_reverse = program(*inputs, cotangent=seed)
            staged_pairing = _real_pairing(
                staged_reverse,
                tangents if len(argnums) > 1 else tangents[0],
            )
            np.testing.assert_allclose(
                staged_pairing,
                forward_pairing,
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"{path}.{parameter} violates the staged serialized VJP identity "
                    f"for Array API {array_api_version}"
                ),
            )
