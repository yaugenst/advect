"""Unit tests for ufunc out= and where= tracing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from numpy.testing import assert_allclose

import advect as ad
from advect.numpy._support_contract import numpy_support_declarations
from advect_numpy_tests._assertions import (
    assert_spellings_agree,
    assert_staged_round_trip,
    assert_tree_close,
)
from advect_numpy_tests._support_case_families import support_cases


class TestUfuncOutParamTracing:
    def test_ufunc_out_rebinds_the_existing_wrapper_to_a_pure_value(self) -> None:
        """out= preserves Python identity while emitting ordinary pure SSA."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)
        observations: list[tuple[bool, int, bool]] = []

        def add_into(x, y):
            z = np.empty_like(x)
            prev_node_id = z.node_id
            res = np.add(x, y, out=z)
            observations.append((res is z, z.epoch, z.node_id != prev_node_id))
            return z

        x_tangent = np.array([0.25, -0.5])
        y_tangent = np.array([1.5, 2.0])
        value, tangent = ad.jvp(add_into, argnums=(0, 1))(
            x0,
            y0,
            tangents=(x_tangent, y_tangent),
        )

        assert observations == [(True, 1, True)]
        assert_allclose(value, x0 + y0)
        assert_allclose(tangent, x_tangent + y_tangent)

    def test_ufunc_out_chains_through_real_data_dependencies_only(self) -> None:
        """Reading the first result makes it the next operation's ordinary input."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)

        observations: list[tuple[int, bool]] = []

        def chained_add(x, y):
            z = np.empty_like(x)
            _ = np.add(x, y, out=z)
            first_write = z.node_id
            _ = np.add(z, 1.0, out=z)
            observations.append((z.epoch, z.node_id != first_write))
            return z

        value, tangent = ad.jvp(chained_add, argnums=(0, 1))(
            x0,
            y0,
            tangents=(np.ones_like(x0), np.ones_like(y0)),
        )

        assert observations == [(2, True)]
        assert_allclose(value, x0 + y0 + 1.0)
        assert_allclose(tangent, np.full_like(x0, 2.0))

    def test_ufunc_out_rejects_non_traced_out(self) -> None:
        """out= must be a TracedArray."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)
        z0 = np.empty_like(x0)

        def add_into_raw(x, y):
            return np.add(x, y, out=z0)

        with pytest.raises(ad.TracingError, match="TracedArray"):
            ad.jvp(add_into_raw, argnums=(0, 1))(
                x0,
                y0,
                tangents=(np.ones_like(x0), np.ones_like(y0)),
            )

    def test_ufunc_where_without_out_is_rejected(self) -> None:
        """where= without out= is rejected during tracing."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)
        mask0 = np.array([True, False], dtype=bool)

        with pytest.raises(ad.TracingError, match="requires out"):
            ad.jvp(lambda x, y: np.add(x, y, where=mask0), argnums=(0, 1))(
                x0,
                y0,
                tangents=(np.ones_like(x0), np.ones_like(y0)),
            )

    def test_ufunc_out_to_integer_has_zero_derivative(self) -> None:
        """Unsafe writes into integer buffers are locally constant."""
        x0 = np.array([0.7, 1.2, -0.4])

        def loss(x):
            destination = np.empty_like(x, dtype=np.int32)
            np.add(x, 0.25, out=destination, casting="unsafe")
            return np.sum(destination.astype(np.float64))

        primal, tangent = ad.jvp(loss)(x0, tangents=np.ones_like(x0))

        expected = np.add(x0, 0.25).astype(np.int32)
        assert_allclose(primal, np.sum(expected))
        assert_allclose(tangent, 0.0)
        assert_allclose(ad.grad(loss)(x0), np.zeros_like(x0))

    @pytest.mark.parametrize(
        ("control", "value"),
        [
            ("dtype", np.float64),
            ("signature", "D->d"),
            ("sig", "D->d"),
        ],
    )
    def test_ufunc_out_rejects_unrepresented_loop_selection(
        self,
        control: str,
        value: object,
    ) -> None:
        """Tracing rejects loop selection rather than recording a wrong derivative."""
        primal = np.array([3 + 4j], dtype=np.complex64)

        def magnitude(x):
            destination = np.empty(x.shape, dtype=np.float64, like=x)
            return np.absolute(x, out=destination, **{control: value})

        reported_control = "signature" if control == "sig" else control
        with pytest.raises(
            ad.TracingError,
            match=rf"{reported_control}=.*loop selection",
        ):
            ad.jvp(magnitude)(primal, tangents=np.ones_like(primal))
        with pytest.raises(ad.TracingError, match=rf"{reported_control}=.*staged out="):
            ad.stage(magnitude, specs=(ad.ArraySpec(primal.shape, primal.dtype),))

    def test_ufunc_dtype_rejection_prevents_post_cast_approximation(self) -> None:
        """Loop precision can differ from post-casting, so tracing rejects it."""
        left = np.array([-10.0], dtype=np.float64)
        right = np.array([-9.74], dtype=np.float64)

        def add_in_float16(x, y):
            destination = np.empty_like(x, dtype=np.float16)
            return np.add(
                x,
                y,
                out=destination,
                dtype=np.float16,
                casting="unsafe",
            )

        expected = np.add(left, right, dtype=np.float16)
        cast_after_default = np.add(left, right).astype(np.float16)
        assert not np.array_equal(expected, cast_after_default)

        with pytest.raises(ad.TracingError, match=r"dtype=.*loop selection"):
            ad.jvp(add_in_float16, argnums=(0, 1))(
                left,
                right,
                tangents=(np.ones_like(left), np.zeros_like(right)),
            )
        with pytest.raises(ad.TracingError, match=r"dtype=.*staged out="):
            ad.stage(
                add_in_float16,
                specs=(
                    ad.ArraySpec(left.shape, left.dtype),
                    ad.ArraySpec(right.shape, right.dtype),
                ),
            )

    def test_ufunc_loop_selection_is_also_rejected_without_out(self) -> None:
        """The correctness boundary applies independently of mutation."""
        value = np.array([0.4, 1.2])

        with pytest.raises(ad.TracingError, match=r"dtype=.*loop selection"):
            ad.jvp(lambda x: np.add(x, x, dtype=np.float32))(
                value,
                tangents=np.ones_like(value),
            )


def test_ufunc_out_rejects_a_destination_view_before_mutation() -> None:
    value = np.asarray([1 + 2j, 3 - 1j])

    def operation(array: Any) -> Any:
        owned = array.copy()
        destination = owned.real
        np.add(destination, 1.0, out=destination)
        return destination

    with pytest.raises(ad.MutationError, match=r"ufunc out=.*traced view"):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


def test_multi_output_ufunc_rejects_out_destinations_explicitly() -> None:
    def operation(array: Any) -> Any:
        fractional = np.zeros_like(array)
        integral = np.zeros_like(array)
        return np.modf(array, out=(fractional, integral))

    with pytest.raises(ad.TracingError, match="Only single-output out="):
        ad.jvp(operation)(np.asarray([1.25, -2.5]), tangents=np.ones(2))


def _log_into(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    del y, mask
    destination = np.zeros_like(x)
    np.log(x, out=destination)
    return destination


def _masked_divide_into(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    destination = np.ones_like(x)
    np.divide(x, y, out=destination, where=mask)
    return destination


@pytest.mark.parametrize("operation", [_log_into, _masked_divide_into], ids=("log", "divide"))
def test_staged_out_validation_raises_no_warnings_for_dummy_values(operation) -> None:
    """Staging validates out= on dummy zeros without reporting their FP warnings."""
    values = (np.array([1.0, 2.0, 4.0]), np.array([2.0, 0.5, 4.0]), np.array([True, False, True]))

    assert_staged_round_trip(operation, *values)


def test_staged_dtype_changing_out_replays_under_derivative_programs() -> None:
    # The cast back to a narrower out= destination must also replay through
    # the portable evaluator that derivative programs use.
    def operation(value: Any, other: Any) -> Any:
        result = value.copy()
        np.add(result, other, out=result)
        return result * result

    value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other = np.array([0.5, -0.25, 2.0])
    expected = operation(value, other)
    program = ad.stage(operation, value, other)
    actual = program(value, other)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)

    cotangent = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    _, dynamic_pullback = ad.vjp(operation)(value, other)
    np.testing.assert_allclose(
        ad.vjp_program(program)(value, other, cotangent=cotangent),
        dynamic_pullback(cotangent),
        rtol=1e-6,
    )


def test_ufunc_out_differentiates_under_an_enclosing_transform() -> None:
    """An enclosing trace sees how ``out=`` depends on the inputs, as without ``out=``."""
    value = np.array([0.1, 0.5, 0.9])
    direction = np.array([1.0, 2.0, 3.0])
    sine, cosine = np.sin(value), np.cos(value)

    def cube(x: Any) -> Any:
        return np.sin(x, out=np.zeros_like(x)) ** 3

    primal, tangent = ad.jvp(lambda x: ad.jvp(cube)(x, tangents=direction)[0])(
        value, tangents=direction
    )
    assert_allclose(primal, sine**3)
    assert_allclose(tangent, 3 * sine**2 * cosine * direction)
    hessian = ad.hessian(lambda x: np.sum(cube(x)))(value)
    assert_allclose(hessian, np.diag(6 * sine * cosine**2 - 3 * sine**3))


_DECLARATIONS = {
    declaration.callable: declaration
    for declaration in numpy_support_declarations()
    if declaration.kind == "ufunc_call" and declaration.has_derivatives
}


def _is_elementwise_single_output(name: str) -> bool:
    # The installed NumPy may predate a declared ufunc, such as matvec before 2.2.
    ufunc = getattr(np, name, None)
    return ufunc is not None and ufunc.nout == 1 and ufunc.signature is None


_WRITABLE_CASES = {
    case.callable.removeprefix("numpy."): case
    for case in support_cases()
    if case.kind == "ufunc_call"
    and case.variant == "baseline"
    and case.callable in _DECLARATIONS
    and _is_elementwise_single_output(case.callable.removeprefix("numpy."))
}
_SINGLE = {"f": np.dtype(np.float32), "c": np.dtype(np.complex64)}
_CONTROLS = st.fixed_dictionaries(
    {},
    optional={"casting": st.just("unsafe"), "order": st.just("K"), "subok": st.just(False)},  # noqa: FBT003
)


@given(
    name=st.sampled_from(sorted(_WRITABLE_CASES)),
    single=st.booleans(),
    traced_destination=st.booleans(),
    mask=st.none() | st.just("traced") | st.lists(st.booleans(), min_size=1, max_size=4),
    controls=_CONTROLS,
)
@example(name="add", single=False, traced_destination=False, mask=[True, False], controls={})
@example(name="add", single=False, traced_destination=False, mask="traced", controls={})
@example(
    name="add",
    single=True,
    traced_destination=False,
    mask=[True, False],
    controls={"casting": "unsafe"},
)
@example(
    name="add",
    single=True,
    traced_destination=True,
    mask=None,
    controls={"casting": "unsafe", "order": "K", "subok": False},
)
@example(
    name="sin",
    single=False,
    traced_destination=False,
    mask=[True, False],
    controls={"casting": "unsafe"},
)
def test_ufunc_out_and_where_write_the_masked_pure_call(
    name: str,
    *,
    single: bool,
    traced_destination: bool,
    mask: str | list[bool] | None,
    controls: dict[str, object],
) -> None:
    """``f(*x, out=d, where=m)`` is ``d`` rebound to ``where(m, f(*x).astype(d.dtype), d)``."""
    case = _WRITABLE_CASES[name]
    ufunc = getattr(np, name)
    values = tuple(np.asarray(spec.data, dtype=spec.dtype) for spec in case.inputs)
    argnums = tuple(index for index, value in enumerate(values) if value.dtype.kind in "fc")
    result = ufunc(*values)
    dtype = _SINGLE[result.dtype.kind] if single else result.dtype
    first = argnums[0]
    threshold = float(np.median(np.real(values[first])))

    def where(inputs: tuple[Any, ...]) -> Any:
        if mask == "traced":
            return np.real(inputs[first]) > threshold
        # A one-element mask broadcasts as a scalar.
        return True if mask is None else np.resize(mask, result.shape if len(mask) > 1 else ())

    def destination(inputs: tuple[Any, ...]) -> Any:
        fill = np.full(result.shape, 7.0, like=inputs[first])
        return (fill * np.real(np.mean(inputs[first])) if traced_destination else fill).astype(
            dtype
        )

    def write(*inputs: Any) -> Any:
        out = destination(inputs)
        masked = {} if mask is None else {"where": where(inputs)}
        assert ufunc(*inputs, out=out, **masked, **controls) is out
        return out

    def pure(*inputs: Any) -> Any:
        return np.where(where(inputs), ufunc(*inputs).astype(dtype), destination(inputs))

    primal = assert_spellings_agree(write, pure, values, argnums=argnums)
    assert_tree_close(primal, write(*values), rtol=0.0)
    if "staged" in _DECLARATIONS[case.callable].modes:
        assert_staged_round_trip(write, *values, rtol=0.0)
