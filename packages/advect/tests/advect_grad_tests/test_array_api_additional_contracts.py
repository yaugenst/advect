"""Additional provider-neutral contracts for the Array API frontend."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api import providers
from advect.core._array_api.frontend import (
    ArrayAPINamespace,
    _accepts_array_api,
    bind_array_api_call,
)
from advect.core._errors import MutationError


def _strict(values: Any, *, dtype: Any = None) -> Any:
    return strict.asarray(values) if dtype is None else strict.asarray(values, dtype=dtype)


def _trace(function: Any, *values: Any) -> Any:
    traced = trace_call(
        function,
        args=values,
        kwargs={},
        argnums=tuple(range(len(values))),
        argnames=None,
    )
    try:
        return traced.output
    finally:
        traced.tape.release_payloads()


def test_call_binding_normalizes_optional_live_parameters() -> None:
    source = object()
    maximum = object()

    clipped = bind_array_api_call("clip", (source,), {"min": None, "max": maximum})
    pinv = bind_array_api_call("linalg.pinv", (source,), {"rtol": None})

    assert clipped.operands == (source, maximum)
    assert clipped.attrs == {
        "_advect_clip_min_is_input": False,
        "_advect_clip_max_is_input": True,
    }
    assert pinv.operands == (source,)
    assert pinv.attrs == {"_advect_pinv_tolerance": None}


@pytest.mark.parametrize(
    ("path", "args", "kwargs", "error", "match"),
    [
        pytest.param(
            "expand_dims",
            (object(), 0),
            {"axis": 1},
            TypeError,
            "received 'axis' twice",
            id="duplicate-positional-attribute",
        ),
        pytest.param(
            "concat",
            (object(),),
            {},
            TypeError,
            "expects 'arrays' to be a list or tuple",
            id="non-sequence-variadic-operand",
        ),
        pytest.param(
            "matrix_transpose",
            (SimpleNamespace(shape=(2,), dtype=np.dtype("float64")),),
            {},
            ValueError,
            "at least two dimensions",
            id="matrix-transpose-rank",
        ),
        pytest.param(
            "future_extension",
            (),
            {},
            NotImplementedError,
            "not traceable yet",
            id="unknown-function",
        ),
    ],
)
def test_call_binding_rejects_invalid_standard_forms(
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        bind_array_api_call(path, args, kwargs)


def test_accumulation_reports_a_missing_provider_dtype() -> None:
    namespace = SimpleNamespace(__name__="minimal", sum=lambda value, **_kwargs: value)
    proxy = ArrayAPINamespace(namespace, array_api_version="2022.12")
    source = SimpleNamespace(shape=(2,), dtype=np.dtype("int32"))

    with pytest.raises(TypeError, match="does not provide dtype 'int64'"):
        proxy.sum(source)


@pytest.mark.parametrize(
    ("provider_result", "error", "match"),
    [
        pytest.param(
            lambda value: [value, value],
            TypeError,
            "must return a tuple of 2 outputs",
            id="wrong-container",
        ),
        pytest.param(
            lambda value: (value,),
            ValueError,
            "returned 1 outputs, expected 2",
            id="wrong-arity",
        ),
        pytest.param(
            lambda value: (value, object()),
            NotImplementedError,
            "object at output 1",
            id="non-array-output",
        ),
    ],
)
def test_multi_output_provider_contract_is_validated_before_recording(
    monkeypatch: pytest.MonkeyPatch,
    provider_result: Any,
    error: type[Exception],
    match: str,
) -> None:
    monkeypatch.setattr(strict.linalg, "eigh", provider_result)
    matrix = _strict([[2.0, 0.0], [0.0, 1.0]], dtype=strict.float64)

    with pytest.raises(error, match=match):
        _trace(lambda value: value.__array_namespace__().linalg.eigh(value), matrix)


def test_single_output_provider_contract_rejects_a_scalar_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strict, "sin", lambda _value: 1.0)
    value = _strict([1.0, 2.0], dtype=strict.float64)

    with pytest.raises(NotImplementedError, match="returned float"):
        _trace(lambda argument: argument.__array_namespace__().sin(argument), value)


def test_dynamic_operand_rejects_an_unrecognized_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strict, "add", lambda left, _right: left)
    value = _strict([1.0, 2.0], dtype=strict.float64)

    with pytest.raises(TypeError, match="Unsupported dynamic Array API operand"):
        _trace(
            lambda argument: argument.__array_namespace__().add(argument, object()),
            value,
        )


@pytest.mark.parametrize(
    ("function", "values", "error", "match"),
    [
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_power(value, bool(1)),
            (_strict([[2.0, 0.0], [0.0, 1.0]], dtype=strict.float64),),
            TypeError,
            "static integer",
            id="matrix-power-bool",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_power(value, 2),
            (_strict([[1.0, 2.0, 3.0]], dtype=strict.float64),),
            ValueError,
            "square matrices",
            id="matrix-power-nonsquare",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_rank(value),
            (_strict([1.0, 2.0], dtype=strict.float64),),
            ValueError,
            "at least two dimensions",
            id="matrix-rank-vector",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().broadcast_arrays(x=value),
            (_strict([1.0, 2.0], dtype=strict.float64),),
            TypeError,
            "one or more arrays",
            id="broadcast-arrays-keyword",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_power(value, 2, 3),
            (_strict([[2.0, 0.0], [0.0, 1.0]], dtype=strict.float64),),
            TypeError,
            "matrix and a static integer exponent",
            id="matrix-power-arity",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().linalg.matrix_rank(value, axis=0),
            (_strict([[2.0, 0.0], [0.0, 1.0]], dtype=strict.float64),),
            TypeError,
            "one array and optional keyword-only rtol",
            id="matrix-rank-keyword",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().meshgrid(value, indexing="invalid"),
            (_strict([1.0, 2.0], dtype=strict.float64),),
            ValueError,
            "indexing must be 'ij' or 'xy'",
            id="meshgrid-indexing",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().meshgrid(value),
            (_strict([[1.0, 2.0]], dtype=strict.float64),),
            ValueError,
            "one-dimensional",
            id="meshgrid-rank",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().meshgrid(value, unexpected=True),
            (_strict([1.0, 2.0], dtype=strict.float64),),
            TypeError,
            "one or more arrays and optional indexing",
            id="meshgrid-keyword",
        ),
        pytest.param(
            lambda value: value.__array_namespace__().unstack(value, value),
            (_strict([[1.0, 2.0]], dtype=strict.float64),),
            TypeError,
            "one array and optional keyword-only axis",
            id="unstack-arity",
        ),
    ],
)
def test_composite_boundaries_fail_before_provider_execution(
    function: Any,
    values: tuple[Any, ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _trace(function, *values)


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        pytest.param(
            lambda argument: argument.__array_namespace__().cumulative_sum(
                argument,
                axis=2,
                include_initial=True,
            ),
            _strict([1.0, 2.0], dtype=strict.float64),
            "axis 2 is out of bounds",
            id="cumulative-axis",
        ),
        pytest.param(
            lambda argument: argument.__array_namespace__().asarray([argument], argument),
            _strict([1.0, 2.0], dtype=strict.float64),
            "expects one positional object argument",
            id="asarray-live-sequence-arity",
        ),
        pytest.param(
            lambda argument: argument.__array_namespace__().diff(
                argument,
                argument,
                prepend=argument,
            ),
            _strict([1.0, 2.0], dtype=strict.float64),
            "expects one positional array argument",
            id="diff-boundary-arity",
        ),
        pytest.param(
            lambda argument: argument.__array_namespace__().unique_values(argument, axis=0),
            _strict([1.0, 2.0], dtype=strict.float64),
            "expects one positional array argument",
            id="dynamic-composite-keyword",
        ),
    ],
)
def test_extended_forms_report_ordinary_boundary_errors(
    function: Any,
    value: Any,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _trace(function, value)


def test_extended_provider_edge_paths() -> None:
    value = _strict([3.0, 1.0, 6.0], dtype=strict.float64)
    unchanged = _trace(
        lambda x: x.__array_namespace__().diff(x, n=0, prepend=x[:1]),
        value,
    )

    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(value))
    with pytest.raises(ValueError, match="require axis="):
        _trace(
            lambda x: x.__array_namespace__().cumulative_sum(x, include_initial=True),
            _strict([[1.0, 2.0]], dtype=strict.float64),
        )
    with pytest.raises(TypeError, match="expects two positional array arguments"):
        _trace(lambda x: x.__array_namespace__().searchsorted(x, sorter=x), value)
    with pytest.raises(ValueError, match="cannot construct an array from a sequence"):
        _trace(lambda x: x.__array_namespace__().asarray([x[0]], copy=False), value)


def test_debug_representation_uses_the_provider_value() -> None:
    representations: list[str] = []
    _trace(
        lambda x: representations.append(repr(x)) or x,
        _strict([1.0], dtype=strict.float64),
    )
    with ad.debug():
        _trace(
            lambda x: representations.append(repr(x)) or x,
            _strict([1.0], dtype=strict.float64),
        )
    assert "values=" not in representations[0]
    assert "values=Array([1.]" in representations[1]


def test_tracer_array_methods_preserve_standard_array_results() -> None:
    value = _strict([1.0, 2.0, 3.0, 4.0], dtype=strict.float32)

    def methods(argument: Any) -> Any:
        namespace = argument.__array_namespace__()
        matrix = argument.reshape(2, 2)
        tuple_reshape = argument.reshape((2, 2))
        return (
            argument.astype(namespace.float64, device=argument.device),
            matrix.T,
            tuple_reshape.mT,
            argument.sum(dtype=namespace.float64),
            argument.item(1),
            matrix.item(0, 1),
            argument[:1].item(),
        )

    (
        converted,
        transposed,
        matrix_transposed,
        total,
        flat_item,
        coordinate_item,
        singleton_item,
    ) = _trace(methods, value)

    assert converted.dtype == strict.float64
    np.testing.assert_array_equal(np.asarray(transposed), [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(np.asarray(matrix_transposed), np.asarray(transposed))
    assert float(np.asarray(total)) == 10.0
    assert float(np.asarray(flat_item)) == 2.0
    assert float(np.asarray(coordinate_item)) == 2.0
    assert float(np.asarray(singleton_item)) == 1.0


def test_tracer_operator_surface_routes_to_standard_operations() -> None:
    real = _strict([1.0, 2.0], dtype=strict.float64)
    integer = _strict([1, 2], dtype=strict.int64)
    complex_value = _strict([1.0 + 2.0j, -3.0 + 0.5j], dtype=strict.complex128)
    matrix = _strict([[2.0, 0.0], [0.0, 3.0]], dtype=strict.float64)

    real_outputs = _trace(
        lambda value: (
            2.0 + value,
            3.0 - value,
            4.0 / value,
            5.0 // value,
            5.0 % value,
            2.0**value,
            value.__rmatmul__(matrix),
            value > 0.0,
            value < 3.0,
            +value,
            -value,
            abs(value),
        ),
        real,
    )
    integer_outputs = _trace(
        lambda value: (1 & value, 1 | value, 1 ^ value, value & 1, value | 1, value ^ 1, ~value),
        integer,
    )
    complex_outputs = _trace(lambda value: (value.conj(), value.real, value.imag), complex_value)

    assert len(real_outputs) == 12
    assert len(integer_outputs) == 7
    np.testing.assert_array_equal(np.asarray(complex_outputs[0]), np.conj(complex_value))
    np.testing.assert_array_equal(np.asarray(complex_outputs[1]), np.real(complex_value))
    np.testing.assert_array_equal(np.asarray(complex_outputs[2]), np.imag(complex_value))


def test_tracer_sequence_scalar_and_mutation_boundaries() -> None:
    value = _strict([1.0, 2.0], dtype=strict.float64)
    scalar = _strict(1.0, dtype=strict.float64)
    observations: list[tuple[int, tuple[tuple[int, ...], ...], bool]] = []

    def inspect(argument: Any) -> Any:
        observations.append(
            (len(argument), tuple(item.shape for item in argument), bool(argument[0]))
        )
        return argument[0]

    _trace(inspect, value)
    assert observations == [(2, ((), ()), True)]

    with pytest.raises(TypeError, match=r"len\(\) of a 0-dimensional array"):
        _trace(len, scalar)
    with pytest.raises(ValueError, match="array of size 1"):
        _trace(lambda argument: argument.item(), value)

    def mutate(argument: Any) -> Any:
        argument[0] = 0.0
        return argument

    with pytest.raises(MutationError, match="generic Array API inputs"):
        _trace(mutate, value)


def test_tracer_rejects_a_different_revision_request() -> None:
    value = _strict([1.0, 2.0], dtype=strict.float64)

    with pytest.raises(ValueError, match=r"provider exposes '2024\.12'"):
        _trace(
            lambda argument: argument.__array_namespace__(api_version="2022.12").sum(argument),
            value,
        )


def test_namespace_proxy_filters_profiles_and_forwards_info() -> None:
    namespace_2022 = ArrayAPINamespace(strict, array_api_version="2022.12")
    namespace_2024 = ArrayAPINamespace(strict, array_api_version="2024.12")

    assert "cumulative_sum" not in dir(namespace_2022)
    assert "cumulative_prod" in dir(namespace_2024)
    assert "svd" in dir(namespace_2024.linalg)
    with pytest.raises(
        AttributeError,
        match=r"not available in the selected 2022\.12 revision",
    ):
        _ = namespace_2022.cumulative_sum
    assert type(namespace_2024.__array_namespace_info__()) is type(
        strict.__array_namespace_info__()
    )


def test_operations_reject_a_namespace_different_from_the_traced_input() -> None:
    value = _strict([1.0, 2.0], dtype=strict.float64)
    alternate = ArrayAPINamespace(
        SimpleNamespace(
            __name__="alternate",
            add=strict.add,
            broadcast_arrays=strict.broadcast_arrays,
        )
    )

    with pytest.raises(TypeError, match="different Array API namespaces in add"):
        _trace(lambda argument: alternate.add(argument, argument), value)
    with pytest.raises(TypeError, match="different Array API namespaces in broadcast_arrays"):
        _trace(alternate.broadcast_arrays, value)


class _ProtocolArray:
    __advect_namespace_is_instance_specific__ = True
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self, namespace: Any) -> None:
        self.namespace = namespace

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        del api_version
        return self.namespace


class _WrappedArray:
    def __init__(self, value: Any) -> None:
        self.value = value

    def _advect_snapshot(self) -> tuple[int, Any]:
        return 1, self.value


def _provider_namespace(
    *,
    name: str | None = "provider",
    version: str | None = "2024.12",
    asarray: bool = True,
    namespace_info: bool = True,
) -> Any:
    attributes: dict[str, Any] = {}
    if name is not None:
        attributes["__name__"] = name
    if version is not None:
        attributes["__array_api_version__"] = version
    if asarray:
        attributes["asarray"] = lambda value: value
    if namespace_info:
        attributes["__array_namespace_info__"] = object
    return SimpleNamespace(**attributes)


def test_provider_namespace_changes_after_negotiation_are_reported() -> None:
    current = _provider_namespace()

    class ChangingArray:
        __advect_namespace_is_instance_specific__ = True
        shape = (1,)
        dtype = np.dtype("float64")

        def __init__(self, replacement: Any) -> None:
            self.calls = 0
            self.replacement = replacement

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            del api_version
            self.calls += 1
            return current if self.calls <= 2 else self.replacement

    with pytest.raises(TypeError, match="no longer exposes an Array API namespace"):
        _trace(lambda value: value, ChangingArray(None))
    with pytest.raises(TypeError, match=r"provider exposes '2022\.12'"):
        _trace(lambda value: value, ChangingArray(_provider_namespace(version="2022.12")))


def test_namespace_discovery_supports_instance_only_and_wrapped_protocols() -> None:
    providers._clear_array_namespace_caches()
    namespace = _provider_namespace()

    class CountingArray:
        calls = 0

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            assert api_version == "2024.12"
            CountingArray.calls += 1
            return namespace

    first = _WrappedArray(_WrappedArray(CountingArray()))
    second = _WrappedArray(_WrappedArray(CountingArray()))
    assert providers._get_array_namespace(first, api_version="2024.12") is namespace
    assert providers._get_array_namespace(second, api_version="2024.12") is namespace
    assert CountingArray.calls == 1

    instance_namespace = _provider_namespace(name="instance-provider")
    wrapped_instance = _WrappedArray(_ProtocolArray(instance_namespace))
    assert (
        providers._get_array_namespace(wrapped_instance, api_version="2024.12")
        is instance_namespace
    )


def test_namespace_discovery_validates_requests_and_empty_wrappers() -> None:
    with pytest.raises(TypeError, match="Invalid Array API version request"):
        providers._get_array_namespace(object(), api_version=object())

    assert providers._get_array_namespace(_WrappedArray(None), api_version="2024.12") is None


@pytest.mark.parametrize(
    "namespace",
    [
        pytest.param(_provider_namespace(name=None), id="missing-backend-name"),
        pytest.param(_provider_namespace(asarray=False), id="missing-asarray"),
        pytest.param(_provider_namespace(namespace_info=False), id="missing-namespace-info"),
        pytest.param(_provider_namespace(version="future"), id="malformed-version"),
    ],
)
def test_negotiation_rejects_incomplete_provider_reports(namespace: Any) -> None:
    value = _ProtocolArray(namespace)

    with pytest.raises(TypeError, match=r"cannot serve required Array API 2024\.12"):
        providers._negotiate_array_namespace_for_call(
            args=(value,),
            kwargs={},
            required_version="2024.12",
        )


def test_acceptance_reports_provider_revision_and_noninvasive_failures() -> None:
    old = _ProtocolArray(_provider_namespace(version="2022.12"))
    with pytest.raises(TypeError, match=r"selected Array API 2024\.12"):
        _accepts_array_api(old)

    class DefaultOnlyArray:
        shape = (1,)
        dtype = np.dtype("float64")

        def __init__(self, *, fail_default: bool = False) -> None:
            self.fail_default = fail_default

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            if api_version is not None or self.fail_default:
                raise ValueError("unsupported request")
            return _provider_namespace()

    assert _accepts_array_api(DefaultOnlyArray())
    assert not _accepts_array_api(DefaultOnlyArray(fail_default=True))
    assert not _accepts_array_api(_ProtocolArray(_provider_namespace(namespace_info=False)))
    assert not _accepts_array_api(SimpleNamespace(shape=(1,), dtype=np.dtype("float64")))
    assert not _accepts_array_api(object())
