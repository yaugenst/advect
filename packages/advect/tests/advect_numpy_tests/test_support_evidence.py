"""Execute every lifetime published by the NumPy support catalog."""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
)
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.numpy._support_contract import numpy_support_declarations
from advect_numpy_tests._assertions import assert_adjoint_identity, seeded_like
from advect_numpy_tests._support_case_families import support_cases
from advect_numpy_tests._support_cases import DType, Function, Input

if TYPE_CHECKING:
    from advect_numpy_tests._support_cases import DerivativeArgnums, NumpySupportCase


def _resolve_callable(path: str) -> Any:
    target: Any = np
    components = path.split(".")
    if not components or components[0] != "numpy":
        raise ValueError(path)
    for component in components[1:]:
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
_DECLARATIONS = {
    (declaration.kind, declaration.callable): declaration
    for declaration in numpy_support_declarations()
}
_ALL_MODES = ("dynamic", "staged", "serialized")
# Dynamic-only lowerings reject staging through these error types.
_STAGING_REJECTIONS = (NotImplementedError, TypeError, ad.TracingError)
_DOUBLE = (np.dtype(np.float64), np.dtype(np.complex128))


def _modes(case: NumpySupportCase) -> tuple[str, ...]:
    return _ALL_MODES if case.stages else _DECLARATIONS[(case.kind, case.callable)].modes


def _derivative_groups(case: NumpySupportCase) -> DerivativeArgnums | None:
    """Return a differentiable case's groups: explicit, or each floating input and all."""
    if not _DECLARATIONS[(case.kind, case.callable)].has_derivatives:
        return None
    if case.derivative_argnums is not None:
        return case.derivative_argnums
    floating = tuple(
        index
        for index, value in enumerate(case.inputs)
        if value.dtype.startswith(("float", "complex"))
    )
    return (*((index,) for index in floating), floating) if len(floating) > 1 else (floating,)


def test_runtime_declarations_have_exact_executable_case_coverage() -> None:
    cases = support_cases()

    assert len(_DECLARATIONS) == len(numpy_support_declarations())
    assert _DECLARATIONS.keys() == {(case.kind, case.callable) for case in cases}
    for case in cases:
        declaration = _DECLARATIONS[(case.kind, case.callable)]
        groups = _derivative_groups(case)
        assert declaration.has_derivatives or case.derivative_argnums is None, case.identifier
        assert groups is None or all(groups), case.identifier
        assert not case.stages or "staged" not in declaration.modes, case.identifier


def test_dynamic_only_forms_fail_closed_under_staging() -> None:
    """A form that stages in every case must declare its staged lifetimes."""
    staging_failures: dict[tuple[str, str], bool] = {}
    for case in _CASES:
        form = (case.kind, case.callable)
        if "staged" in _modes(case) or staging_failures.get(form):
            continue
        inputs = _materialize_inputs(case)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Example inputs select an Array API revision every NumPy minor serves.
                ad.stage(lambda *values, case=case: _invoke(case, values), *inputs)(*inputs)
        except _STAGING_REJECTIONS:
            staging_failures[form] = True
        except AttributeError:
            # Only a method is missing from the abstract array staging traces.
            if case.kind != "array_method":
                raise
            staging_failures[form] = True
        else:
            staging_failures.setdefault(form, False)

    assert not sorted(form for form, failed in staging_failures.items() if not failed)


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
    if case.result_adapter == "fields":
        return dict(result._asdict())
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


def _assert_tangent_structure(primal: Any, tangent: Any) -> None:
    """Match the primal structure, with zero tangents on non-inexact leaves."""
    primal_leaves, primal_treedef = tree_flatten(primal)
    tangent_leaves, tangent_treedef = tree_flatten(tangent)
    assert tangent_treedef == primal_treedef
    for primal_leaf, tangent_leaf in zip(primal_leaves, tangent_leaves, strict=True):
        primal_array, tangent_array = np.asarray(primal_leaf), np.asarray(tangent_leaf)
        assert tangent_array.shape == primal_array.shape
        assert tangent_array.dtype.kind in "fc"
        assert primal_array.dtype.kind in "fc" or not np.any(tangent_array)


def _assert_central_difference(
    call: Any,
    inputs: tuple[np.ndarray[Any, Any], ...],
    argnums: tuple[int, ...],
    directions: tuple[np.ndarray[Any, Any], ...],
    tangent: Any,
) -> None:
    """Compare double-precision tangent leaves with a central difference.

    Single precision cannot resolve the step, and an output whose structure,
    shape or dtype moves with the input (such as unique values) has no
    difference quotient; the adjoint identity still covers both.
    """
    if any(inputs[index].dtype not in _DOUBLE for index in argnums):
        return
    step = 1e-6

    def shifted(sign: float) -> tuple[list[Any], Any]:
        values = list(inputs)
        for index, direction in zip(argnums, directions, strict=True):
            values[index] = inputs[index] + sign * step * direction
        return tree_flatten(call(*values))

    tangent_leaves, tangent_treedef = tree_flatten(tangent)
    (upper_leaves, upper_treedef), (lower_leaves, lower_treedef) = shifted(1.0), shifted(-1.0)
    if not tangent_treedef == upper_treedef == lower_treedef:
        return
    for leaf, upper, lower in zip(tangent_leaves, upper_leaves, lower_leaves, strict=True):
        upper_array, lower_array = np.asarray(upper), np.asarray(lower)
        if (
            upper_array.dtype == lower_array.dtype == np.asarray(leaf).dtype
            and upper_array.dtype in _DOUBLE
            and upper_array.shape == lower_array.shape == np.shape(leaf)
        ):
            np.testing.assert_allclose(
                leaf, (upper_array - lower_array) / (2 * step), rtol=1e-5, atol=1e-6
            )


class _DerivativeContract(NamedTuple):
    """One derivative group's seeded directions and cotangent with dynamic results."""

    argnums: tuple[int, ...]
    directions: tuple[np.ndarray[Any, Any], ...]
    tangent: Any
    seed: Any
    cotangents: tuple[Any, ...]


def _qualify_dynamic_derivatives(
    case: NumpySupportCase,
    groups: DerivativeArgnums,
    call: Any,
    inputs: tuple[np.ndarray[Any, Any], ...],
    expected: Any,
) -> tuple[_DerivativeContract, ...]:
    contracts = []
    for argnums in groups:
        salt = f"{case.identifier}:{argnums}"
        directions = tuple(seeded_like(inputs[index], f"{salt}:{index}") for index in argnums)
        value, tangent = ad.jvp(call, argnums)(*inputs, tangents=directions)
        _assert_metadata_and_values(value, expected, compare_values=case.compare_values)
        _assert_tangent_structure(value, tangent)
        _assert_central_difference(call, inputs, argnums, directions, tangent)
        value, seed, cotangents = assert_adjoint_identity(
            call, inputs, directions, tangent, argnums=argnums, salt=salt
        )
        _assert_metadata_and_values(value, expected, compare_values=case.compare_values)
        contracts.append(_DerivativeContract(argnums, directions, tangent, seed, cotangents))
    return tuple(contracts)


def _assert_no_derivatives(
    case: NumpySupportCase,
    call: Any,
    inputs: tuple[np.ndarray[Any, Any], ...],
    expected: Any,
) -> None:
    """Push seeded directions of every inexact input through a form without derivatives.

    Its outputs are discrete or locally constant, so every tangent is zero.
    """
    argnums = tuple(index for index, value in enumerate(inputs) if value.dtype.kind in "fc")
    if not argnums:
        return
    salt = f"{case.identifier}:no-derivatives"
    directions = tuple(seeded_like(inputs[index], f"{salt}:{index}") for index in argnums)
    value, tangent = ad.jvp(call, argnums)(*inputs, tangents=directions)
    _assert_metadata_and_values(value, expected, compare_values=case.compare_values)
    primal_leaves, primal_treedef = tree_flatten(value)
    tangent_leaves, tangent_treedef = tree_flatten(tangent)
    assert tangent_treedef == primal_treedef
    for primal_leaf, tangent_leaf in zip(primal_leaves, tangent_leaves, strict=True):
        assert np.shape(tangent_leaf) == np.shape(primal_leaf)
        assert not np.any(np.asarray(tangent_leaf))


def _qualify_staged_derivatives(
    program: ad.StagedProgram,
    inputs: tuple[np.ndarray[Any, Any], ...],
    snapshots: tuple[np.ndarray[Any, Any], ...],
    contracts: tuple[_DerivativeContract, ...],
) -> None:
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for contract in contracts:
        _, tangent = ad.jvp(restored, contract.argnums)(*inputs, tangents=contract.directions)
        _assert_metadata_and_values(tangent, contract.tangent, compare_values=True)
        _assert_inputs_unchanged(inputs, snapshots)
        pullback_program = ad.vjp_program(restored, argnums=contract.argnums)
        serialized_pullback = ad.StagedProgram.from_dict(pullback_program.to_dict())
        for staged_pullback in (pullback_program, serialized_pullback):
            cotangents = staged_pullback(*inputs, cotangent=contract.seed)
            _assert_metadata_and_values(cotangents, contract.cotangents, compare_values=True)
            _assert_inputs_unchanged(inputs, snapshots)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.identifier)
def test_published_numpy_lifetimes_execute(case: NumpySupportCase) -> None:
    inputs = _materialize_inputs(case)
    snapshots = tuple(np.array(value, copy=True) for value in inputs)
    expected = _invoke(case, tuple(np.array(value, copy=True) for value in inputs))

    def call(*values: Any) -> Any:
        return _invoke(case, values)

    trace_indices = tuple(
        index for index, value in enumerate(inputs) if value.dtype.kind in "fc"
    ) or tuple(range(len(inputs)))
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
    modes = _modes(case)
    if "staged" in modes:
        specs = tuple(ad.ArraySpec(value.shape, value.dtype) for value in inputs)
        program = ad.stage(
            call,
            specs=specs,
            array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
        )
        outputs["staged"] = program(*inputs)
        _assert_inputs_unchanged(inputs, snapshots)
        restored = ad.StagedProgram.from_dict(program.to_dict())
        outputs["serialized"] = restored(*inputs)
        _assert_inputs_unchanged(inputs, snapshots)

    assert set(outputs) == set(modes)
    for output in outputs.values():
        _assert_metadata_and_values(output, expected, compare_values=case.compare_values)

    groups = _derivative_groups(case)
    if groups is None:
        _assert_no_derivatives(case, call, inputs, expected)
        _assert_inputs_unchanged(inputs, snapshots)
    else:
        derivative_contracts = _qualify_dynamic_derivatives(case, groups, call, inputs, expected)
        _assert_inputs_unchanged(inputs, snapshots)
        if program is not None:
            _qualify_staged_derivatives(
                program,
                inputs,
                snapshots,
                derivative_contracts,
            )


_OLDEST_ARRAY_API_VERSION = SUPPORTED_ARRAY_API_VERSIONS[0]
_STAGED_DERIVATIVE_CASES = tuple(
    case for case in _CASES if "staged" in _modes(case) and _derivative_groups(case) is not None
)
# A dynamic trace nested in staging wraps its inputs as Array API tracers,
# whose methods accept only the Array API's arguments.
_NUMPY_METHOD_ARGUMENTS = frozenset(
    {
        "array_method:numpy.ndarray.astype[controls]",
        "array_method:numpy.ndarray.sum[controls]",
        "array_method:numpy.ndarray.sum[positional-axis]",
    }
)


def _derivatives_at_the_oldest_revision(
    case: NumpySupportCase,
) -> tuple[Any, tuple[np.ndarray[Any, Any], ...], ad.StagedProgram, DerivativeArgnums]:
    """Return a case's call, inputs, its program at the oldest revision, and its groups.

    A NumPy program may use operations newer than its Array API target, which
    NumPy 2.0 selects by default. Its derivative rules and replayed derivative
    graphs must not, or they fail to stage at that target.
    """
    inputs = _materialize_inputs(case)
    groups = _derivative_groups(case)
    assert groups is not None

    def call(*values: Any) -> Any:
        return _invoke(case, values)

    program = ad.stage(call, *inputs, array_api_version=_OLDEST_ARRAY_API_VERSION)
    return call, inputs, program, groups


@pytest.mark.parametrize("case", _STAGED_DERIVATIVE_CASES, ids=lambda case: case.identifier)
def test_pullback_programs_restage_at_the_oldest_revision(case: NumpySupportCase) -> None:
    call, inputs, program, groups = _derivatives_at_the_oldest_revision(case)
    count = len(inputs)
    for argnums in groups:
        salt = f"{case.identifier}:{argnums}"
        directions = tuple(seeded_like(inputs[index], f"{salt}:{index}") for index in argnums)
        _, tangent = ad.jvp(call, argnums)(*inputs, tangents=directions)
        _, seed, cotangents = assert_adjoint_identity(
            call, inputs, directions, tangent, argnums=argnums, salt=salt
        )
        seeds, treedef = tree_flatten(seed)
        pullback = ad.vjp_program(program, argnums=argnums)
        restaged = ad.stage(
            lambda *values, pullback=pullback, treedef=treedef: pullback(
                *values[:count], cotangent=tree_unflatten(treedef, list(values[count:]))
            ),
            *inputs,
            *seeds,
            array_api_version=_OLDEST_ARRAY_API_VERSION,
        )
        for actual in (pullback(*inputs, cotangent=seed), restaged(*inputs, *seeds)):
            _assert_metadata_and_values(actual, cotangents, compare_values=True)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.xfail(raises=TypeError, reason="NumPy-only method arguments"),
        )
        if case.identifier in _NUMPY_METHOD_ARGUMENTS
        else case
        for case in _STAGED_DERIVATIVE_CASES
    ],
    ids=lambda case: case.identifier,
)
def test_forward_mode_stages_at_the_oldest_revision(case: NumpySupportCase) -> None:
    call, inputs, program, groups = _derivatives_at_the_oldest_revision(case)
    count = len(inputs)
    for argnums in groups:
        salt = f"{case.identifier}:{argnums}"
        directions = tuple(seeded_like(inputs[index], f"{salt}:{index}") for index in argnums)
        _, tangent = ad.jvp(call, argnums)(*inputs, tangents=directions)
        for function in (call, program):
            forward = ad.stage(
                lambda *values, function=function, argnums=argnums: ad.jvp(function, argnums)(
                    *values[:count], tangents=values[count:]
                )[1],
                *inputs,
                *directions,
                array_api_version=_OLDEST_ARRAY_API_VERSION,
            )
            _assert_metadata_and_values(forward(*inputs, *directions), tangent, compare_values=True)


def _positional_spelling(case: NumpySupportCase) -> NumpySupportCase | None:
    """Pass a staged function's keyword metadata by position where NumPy allows it."""
    if case.kind != "function" or "staged" not in _modes(case):
        return None
    try:
        parameters = inspect.signature(_resolve_callable(case.callable)).parameters.values()
    except (TypeError, ValueError):
        return None
    args, keywords = list(case.args), dict(case.kwargs)
    for parameter in list(parameters)[len(args) :]:
        if not keywords or parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            break
        if parameter.name not in keywords and parameter.default is not None:
            break
        args.append(keywords.pop(parameter.name, None))
    if len(keywords) == len(case.kwargs):
        return None
    return dataclasses.replace(case, args=tuple(args), kwargs=tuple(keywords.items()))


_POSITIONAL_SPELLINGS = tuple(
    (case, spelling) for case in _CASES if (spelling := _positional_spelling(case)) is not None
)


@pytest.mark.parametrize(
    ("case", "spelling"),
    _POSITIONAL_SPELLINGS,
    ids=[case.identifier for case, _spelling in _POSITIONAL_SPELLINGS],
)
def test_staging_binds_positional_metadata_as_numpy_does(
    case: NumpySupportCase,
    spelling: NumpySupportCase,
) -> None:
    inputs = _materialize_inputs(case)
    staged = ad.stage(lambda *values: _invoke(spelling, values), *inputs)(*inputs)

    _assert_metadata_and_values(
        staged,
        _invoke(case, inputs),
        compare_values=case.compare_values,
    )


_STAGED_CASES = tuple(case for case in _CASES if "staged" in _modes(case))


@pytest.mark.parametrize("case", _STAGED_CASES, ids=lambda case: case.identifier)
def test_staging_lowers_the_same_graph_from_another_providers_examples(
    case: NumpySupportCase,
) -> None:
    # Staged code sees the examples' provider dtype objects, which NumPy
    # cannot interpret, so the frontend lowers from the staged dtypes alone.
    inputs = _materialize_inputs(case)
    version = min(np.__array_api_version__, LATEST_ARRAY_API_VERSION)

    def graph(*examples: object) -> object:
        program = ad.stage(
            lambda *values: _invoke(case, values), *examples, array_api_version=version
        )
        return program.to_dict()["program"]["graph"]

    assert graph(*(strict.asarray(value) for value in inputs)) == graph(*inputs)


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_staged_reduction_methods_bind_positional_metadata(method: str) -> None:
    value = np.arange(12.0).reshape(3, 4)

    def reduce(x: Any) -> Any:
        return getattr(x, method)(1, np.float32, None, True)  # noqa: FBT003

    _assert_metadata_and_values(ad.stage(reduce, value)(value), reduce(value), compare_values=True)


def test_catalog_projects_the_available_runtime_declarations() -> None:
    rows = ad.support_catalog()["extensions"]["numpy"]["functions"]
    projected = {
        (str(row["kind"]), str(row["callable"])): (
            tuple(mode for mode in _ALL_MODES if row[mode]),
            row["jvp"] in {"yes", "composite"},
        )
        for row in rows
    }

    assert projected == {
        form: (_DECLARATIONS[form].modes, _DECLARATIONS[form].has_derivatives)
        for form in {(case.kind, case.callable) for case in _CASES}
    }
