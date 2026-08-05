"""Unit tests for ufunc where= tracing (restricted to out=)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


class TestUfuncWhereParamTracing:
    def test_ufunc_out_where_traces_and_executes(self) -> None:
        """out= + where= traces and replays correctly."""
        x0 = np.array([-1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)
        mask0 = np.array([True, False], dtype=bool)
        observations: list[tuple[bool, bool]] = []

        def masked_add(x, y):
            z = np.zeros_like(x)
            prev_node_id = z.node_id
            res = np.add(x, y, out=z, where=mask0)
            observations.append((res is z, z.node_id != prev_node_id))
            return z

        expected = np.zeros_like(x0)
        np.add(x0, y0, out=expected, where=mask0)
        x_tangent = np.array([0.25, -0.5])
        y_tangent = np.array([1.5, 2.0])
        value, tangent = ad.jvp(masked_add, argnums=(0, 1))(
            x0,
            y0,
            tangents=(x_tangent, y_tangent),
        )

        assert observations == [(True, True)]
        assert_allclose(value, expected)
        assert_allclose(tangent, np.where(mask0, x_tangent + y_tangent, 0.0))

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

    def test_ufunc_out_where_composes_with_casting_controls(self) -> None:
        """where= remains differentiable when standard ufunc controls are present."""
        x0 = np.array([1.0, 2.0], dtype=np.float64)
        y0 = np.array([3.0, 4.0], dtype=np.float64)

        def add_with_casting(x, y):
            z = np.zeros_like(x, dtype=np.float32)
            return np.add(
                x,
                y,
                out=z,
                where=np.array([True, False]),
                casting="unsafe",
            )

        value, tangent = ad.jvp(add_with_casting, argnums=(0, 1))(
            x0,
            y0,
            tangents=(np.ones_like(x0), np.ones_like(y0)),
        )

        assert value.dtype == np.dtype(np.float32)
        assert_allclose(value, np.array([4.0, 0.0], dtype=np.float32))
        assert_allclose(tangent, np.array([2.0, 0.0], dtype=np.float32))

    def test_ufunc_out_where_accepts_traced_mask(self) -> None:
        """where= accepts traced boolean arrays."""
        x0 = np.array([-1.0, 0.5, 2.0], dtype=np.float64)

        def masked_add(x):
            z = np.zeros_like(x)
            mask = x > 0
            _ = np.add(x, 1.0, out=z, where=mask)
            return z

        expected = np.zeros_like(x0)
        np.add(x0, 1.0, out=expected, where=(x0 > 0))
        tangent_in = np.array([0.25, -0.5, 2.0])
        value, tangent = ad.jvp(masked_add)(x0, tangents=tangent_in)

        assert_allclose(value, expected)
        assert_allclose(tangent, np.where(x0 > 0, tangent_in, 0.0))

    def test_ufunc_out_where_masks_reverse_mode_contributions(self) -> None:
        """The masked destination is an SSA input, not an ignored VJP control."""
        x0 = np.array([1.2, 2.3], dtype=np.float64)
        mask = np.array([True, False])

        def loss(x):
            destination = np.ones_like(x) * 7
            np.sin(x, out=destination, where=mask, casting="unsafe")
            return np.sum(destination)

        assert_allclose(
            ad.grad(loss)(x0),
            np.where(mask, np.cos(x0), 0.0),
        )
