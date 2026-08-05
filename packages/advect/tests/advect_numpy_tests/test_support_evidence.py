"""Execute every lifetime published by the NumPy support catalog."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._pytree import tree_flatten, tree_map
from advect.numpy._support_contract import numpy_support_declarations
from advect_numpy_tests._support_cases import (
    DType,
    Function,
    Input,
    support_cases,
)

if TYPE_CHECKING:
    from advect_numpy_tests._support_cases import NumpySupportCase


def _resolve_callable(path: str) -> Any:
    target: Any = np
    components = path.split(".")
    if not components or components[0] != "numpy":
        raise ValueError(path)
    for component in components[1:]:
        if component == "ndarray":
            continue
        target = getattr(target, component)
    return target


def _callable_exists(case: NumpySupportCase) -> bool:
    if case.kind == "array_method":
        return hasattr(np.ndarray, case.callable.rsplit(".", 1)[-1])
    try:
        _resolve_callable(case.callable)
    except AttributeError:
        return False
    return True


_CASES = tuple(case for case in support_cases() if _callable_exists(case))


def test_runtime_declarations_have_exact_executable_case_coverage() -> None:
    declaration_items = numpy_support_declarations()
    declarations = {
        (declaration.kind, declaration.callable): declaration for declaration in declaration_items
    }
    cases_by_form: dict[tuple[str, str], list[NumpySupportCase]] = {}
    for case in support_cases():
        cases_by_form.setdefault((case.kind, case.callable), []).append(case)

    assert len(declarations) == len(declaration_items)
    assert declarations.keys() == cases_by_form.keys()
    for form, cases in cases_by_form.items():
        modes = set(cases[0].modes)
        derivative_statuses = {case.derivative_argnums is not None for case in cases}
        for case in cases[1:]:
            modes.intersection_update(case.modes)
        declaration = declarations[form]
        assert set(declaration.modes) == modes
        assert derivative_statuses == {declaration.has_derivatives}


def _materialize_inputs(case: NumpySupportCase) -> tuple[np.ndarray[Any, Any], ...]:
    return tuple(np.asarray(spec.data, dtype=np.dtype(spec.dtype)) for spec in case.inputs)


def _resolve(value: object, inputs: tuple[Any, ...]) -> object:
    if isinstance(value, Input):
        return inputs[value.index]
    if isinstance(value, DType):
        return np.dtype(value.name)
    if isinstance(value, Function):
        return _resolve_callable(f"numpy.{value.path}")
    if isinstance(value, tuple):
        return tuple(_resolve(item, inputs) for item in value)
    if isinstance(value, list):
        return [_resolve(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, inputs) for key, item in value.items()}
    return value


def _invoke(case: NumpySupportCase, inputs: tuple[Any, ...]) -> Any:
    working_inputs = inputs
    if case.return_input is not None:
        mutable_inputs = list(inputs)
        mutable_inputs[case.return_input] = np.copy(inputs[case.return_input])
        working_inputs = tuple(mutable_inputs)
    arguments = tuple(_resolve(value, working_inputs) for value in case.args)
    keywords = {name: _resolve(value, working_inputs) for name, value in case.kwargs}
    if case.kind == "array_method":
        target = getattr(working_inputs[0], case.callable.rsplit(".", 1)[-1])
    else:
        target = _resolve_callable(case.callable)
        if case.kind == "ufunc_method":
            arguments = (inputs[0], *arguments)
    with warnings.catch_warnings():
        if case.expected_deprecation is not None:
            warnings.filterwarnings(
                "ignore",
                message=case.expected_deprecation,
                category=DeprecationWarning,
            )
        result = target(*arguments, **keywords)
    result = working_inputs[case.return_input] if case.return_input is not None else result
    if case.result_adapter == "array":
        return np.asarray(result)
    if case.result_adapter == "dtype_num":
        return np.asarray(np.dtype(result).num)
    if case.result_adapter == "tuple":
        return tuple(result)
    return result


def _assert_metadata_and_values(
    actual: Any,
    expected: Any,
    *,
    compare_values: bool,
) -> None:
    actual_leaves, actual_treedef = tree_flatten(actual)
    expected_leaves, expected_treedef = tree_flatten(expected)
    assert actual_treedef == expected_treedef
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        if compare_values:
            np.testing.assert_allclose(actual_array, expected_array, equal_nan=True)


def _assert_inputs_unchanged(
    inputs: tuple[np.ndarray[Any, Any], ...],
    snapshots: tuple[np.ndarray[Any, Any], ...],
) -> None:
    for value, snapshot in zip(inputs, snapshots, strict=True):
        assert value.dtype == snapshot.dtype
        np.testing.assert_array_equal(value, snapshot, strict=True)


def _direction_like(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    direction = np.ones_like(value)
    if value.dtype.kind == "c":
        direction *= 1.0 + 1.0j
    return direction


def _assert_tangent_structure(primal: Any, tangent: Any) -> None:
    primal_leaves, primal_treedef = tree_flatten(primal)
    tangent_leaves, tangent_treedef = tree_flatten(tangent)
    assert tangent_treedef == primal_treedef
    assert all(
        np.shape(tangent_leaf) == np.shape(primal_leaf)
        and np.asarray(tangent_leaf).dtype.kind in {"f", "c"}
        for primal_leaf, tangent_leaf in zip(primal_leaves, tangent_leaves, strict=True)
    )


def _seed_like(tangent: Any) -> Any:
    return tree_map(lambda leaf: np.ones_like(np.asarray(leaf)), tangent)


def _assert_cotangent_metadata(
    cotangents: Any,
    inputs: tuple[np.ndarray[Any, Any], ...],
    indices: tuple[int, ...],
) -> None:
    expected = tuple(np.zeros_like(inputs[index]) for index in indices)
    _assert_metadata_and_values(cotangents, expected, compare_values=False)


def _qualify_dynamic_derivatives(
    case: NumpySupportCase,
    call: Any,
    inputs: tuple[np.ndarray[Any, Any], ...],
    expected: Any,
) -> tuple[tuple[tuple[int, ...], Any, Any], ...]:
    groups = case.derivative_argnums
    assert groups is not None
    contracts = []
    for argnums in groups:
        directions = tuple(_direction_like(inputs[index]) for index in argnums)
        value, tangent = ad.jvp(call, argnums)(*inputs, tangents=directions)
        _assert_metadata_and_values(value, expected, compare_values=case.compare_values)
        _assert_tangent_structure(value, tangent)
        seed = _seed_like(tangent)
        value, pullback = ad.vjp(call, argnums)(*inputs)
        try:
            cotangents = pullback(seed)
        finally:
            pullback.close()
        _assert_metadata_and_values(value, expected, compare_values=case.compare_values)
        _assert_cotangent_metadata(cotangents, inputs, argnums)
        contracts.append((argnums, seed, cotangents))
    return tuple(contracts)


def _qualify_staged_derivatives(
    program: ad.StagedProgram,
    inputs: tuple[np.ndarray[Any, Any], ...],
    snapshots: tuple[np.ndarray[Any, Any], ...],
    contracts: tuple[tuple[tuple[int, ...], Any, Any], ...],
) -> None:
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for argnums, seed, dynamic_cotangents in contracts:
        pullback_program = ad.vjp_program(restored, argnums=argnums)
        serialized_pullback = ad.StagedProgram.from_dict(pullback_program.to_dict())
        for staged_pullback in (pullback_program, serialized_pullback):
            cotangents = staged_pullback(*inputs, cotangent=seed)
            _assert_metadata_and_values(cotangents, dynamic_cotangents, compare_values=True)
            _assert_cotangent_metadata(cotangents, inputs, argnums)
            _assert_inputs_unchanged(inputs, snapshots)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.identifier)
def test_published_numpy_lifetimes_execute(case: NumpySupportCase) -> None:
    inputs = _materialize_inputs(case)
    snapshots = tuple(np.array(value, copy=True) for value in inputs)
    expected = _invoke(case, tuple(np.array(value, copy=True) for value in inputs))

    def call(*values: Any) -> Any:
        return _invoke(case, values)

    trace_indices = case.trace_argnums
    if trace_indices is None:
        trace_indices = tuple(
            index for index, value in enumerate(inputs) if value.dtype.kind in {"f", "c"}
        )
    if not trace_indices:
        trace_indices = tuple(range(len(inputs)))
    assert trace_indices, f"{case.identifier} needs a tracer anchor"
    trace = trace_call(
        call,
        args=inputs,
        kwargs={},
        argnums=trace_indices,
        argnames=None,
    )
    try:
        dynamic = trace.output
    finally:
        trace.tape.release_payloads()
    _assert_metadata_and_values(dynamic, expected, compare_values=case.compare_values)
    _assert_inputs_unchanged(inputs, snapshots)

    outputs = {"dynamic": dynamic}
    program: ad.StagedProgram | None = None
    if "staged" in case.modes:
        specs = tuple(ad.ArraySpec(value.shape, value.dtype) for value in inputs)
        program = ad.stage(
            call,
            specs=specs,
            array_api_version=np.__array_api_version__,
        )
        outputs["staged"] = program(*inputs)
        _assert_inputs_unchanged(inputs, snapshots)
        restored = ad.StagedProgram.from_dict(program.to_dict())
        outputs["serialized"] = restored(*inputs)
        _assert_inputs_unchanged(inputs, snapshots)

    assert set(outputs) == set(case.modes)
    for output in outputs.values():
        _assert_metadata_and_values(output, expected, compare_values=case.compare_values)

    if case.derivative_argnums is not None:
        derivative_contracts = _qualify_dynamic_derivatives(case, call, inputs, expected)
        _assert_inputs_unchanged(inputs, snapshots)
        if program is not None:
            _qualify_staged_derivatives(
                program,
                inputs,
                snapshots,
                derivative_contracts,
            )


def test_catalog_modes_are_exactly_the_available_executable_contract() -> None:
    rows = ad.support_catalog()["extensions"]["numpy"]["functions"]
    rows_by_form = {(str(row["kind"]), str(row["callable"])): row for row in rows}
    cases_by_form: dict[tuple[str, str], list[NumpySupportCase]] = {}
    for case in _CASES:
        cases_by_form.setdefault((case.kind, case.callable), []).append(case)

    assert rows_by_form.keys() == cases_by_form.keys()
    for form, cases in cases_by_form.items():
        modes = set(cases[0].modes)
        for case in cases[1:]:
            modes.intersection_update(case.modes)
        row = rows_by_form[form]
        assert {mode for mode in ("dynamic", "staged", "serialized") if row[mode]} == modes


def test_derivative_roles_are_exactly_the_catalog_capabilities() -> None:
    rows = ad.support_catalog()["extensions"]["numpy"]["functions"]
    rows_by_form = {(str(row["kind"]), str(row["callable"])): row for row in rows}
    inconsistent = sorted(
        case.identifier
        for case in _CASES
        if (case.derivative_argnums is not None)
        != (rows_by_form[(case.kind, case.callable)]["jvp"] in {"yes", "composite"})
    )
    assert not inconsistent, (
        f"NumPy support cases have inconsistent derivative roles: {inconsistent}"
    )
