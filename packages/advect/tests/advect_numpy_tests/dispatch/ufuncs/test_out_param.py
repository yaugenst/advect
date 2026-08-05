"""Unit tests for ufunc out= tracing."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


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

    def test_ufunc_out_supports_standard_execution_kwargs(self) -> None:
        """Functionalized out= preserves NumPy casting and output dtype."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)

        def add_with_casting(x, y):
            z = np.empty_like(x, dtype=np.float32)
            return np.add(
                x,
                y,
                out=z,
                casting="unsafe",
                order="K",
                subok=False,
            )

        value, tangent = ad.jvp(add_with_casting, argnums=(0, 1))(
            x0,
            y0,
            tangents=(np.ones_like(x0), np.ones_like(y0)),
        )

        assert value.dtype == np.dtype(np.float32)
        assert tangent.dtype == np.dtype(np.float32)
        assert_allclose(value, x0 + y0)
        assert_allclose(tangent, np.full_like(x0, 2.0))

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
