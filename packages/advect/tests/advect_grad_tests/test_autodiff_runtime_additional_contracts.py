"""Additional public contracts for dynamic transform boundaries."""

from __future__ import annotations

import numpy as np
import pytest

import advect as ad


class _NonScalarNamespace:
    __name__ = "tests_non_scalar_provider"
    __array_api_version__ = "2024.12"
    float64 = np.float64

    @staticmethod
    def __array_namespace_info__() -> object:
        return object()

    @staticmethod
    def asarray(value: object, *, dtype: object | None = None) -> np.ndarray:
        return np.atleast_1d(np.asarray(value, dtype=dtype))


class _NonScalarArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __array_namespace__(
        self,
        *,
        api_version: str | None = None,
    ) -> type[_NonScalarNamespace]:
        del api_version
        return _NonScalarNamespace


class _FlakyNamespace(_NonScalarNamespace):
    __name__ = "tests_flaky_provider"

    @staticmethod
    def asarray(value: object, *, dtype: object | None = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)


class _FlakyArray:
    __advect_namespace_is_instance_specific__ = True
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self) -> None:
        self.calls = 0

    def __array_namespace__(
        self,
        *,
        api_version: str | None = None,
    ) -> type[_FlakyNamespace] | None:
        del api_version
        self.calls += 1
        return _FlakyNamespace if self.calls % 2 else None


def test_named_selection_reports_an_uninspectable_callable() -> None:
    with pytest.raises(ValueError, match=r"Cannot inspect signature.*max"):
        ad.grad(max, argnums=None, argnames=("value",))


def test_uninspectable_callable_uses_a_stable_fallback_input_label() -> None:
    with pytest.raises(TypeError, match=r"arg0\['coefficient'\].*Python complex scalar"):
        ad.grad(max)({"coefficient": 1.0 + 2.0j}, 3.0)


def test_nested_named_keyword_selection_keeps_the_outer_trace_passive() -> None:
    inner = ad.grad(
        lambda value, *, scale: value * scale,
        argnums=(),
        argnames=("scale",),
    )
    outer = ad.grad(lambda scale: inner(2.0, scale=scale)["scale"])

    assert inner(2.0, scale=3.0) == {"scale": pytest.approx(2.0)}
    assert outer(3.0) == pytest.approx(0.0)


def test_default_gradient_selection_rejects_a_zero_argument_call() -> None:
    with pytest.raises(IndexError, match="index 0 is out of range for 0"):
        ad.grad(lambda: 3.0)()


def test_scalar_lifting_rejects_a_provider_that_returns_a_vector() -> None:
    with pytest.raises(TypeError, match="did not produce a rank-zero scalar primal"):
        ad.grad(lambda tree: tree["scalar"])({"scalar": 2.0, "provider_anchor": _NonScalarArray()})


def test_dynamic_transform_rejects_an_unstable_array_namespace() -> None:
    value = _FlakyArray()

    with pytest.raises(TypeError, match=r"requires Array API 2024\.12"):
        ad.grad(lambda item: item)(value)


def test_staged_gradient_rejects_a_selected_static_leaf_at_its_boundary() -> None:
    program = ad.stage(
        lambda tree: tree["value"] * tree["value"],
        specs=(
            {
                "value": ad.ArraySpec((), "float64"),
                "label": ad.StaticSpec("fixed"),
            },
        ),
    )

    with pytest.raises(TypeError, match="cannot select static input leaves"):
        ad.grad(program)


def test_implicit_root_reports_a_mismatched_residual_structure() -> None:
    def residual(solution: object, params: object) -> object:
        value = solution - params  # type: ignore[operator]
        if isinstance(solution, np.ndarray):
            return value
        return {"value": value}

    root = ad.implicit_root(
        residual,
        solve=lambda residual_at_params, initial: initial - residual_at_params(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )

    with pytest.raises(TypeError, match="implicit residual pytree structure"):
        ad.jvp(root)(
            np.ones(2),
            initial=np.zeros(2),
            tangents=np.ones(2),
        )


def test_implicit_root_preserves_boolean_scalar_specs_for_direct_calls() -> None:
    root = ad.implicit_root(
        lambda solution, _params: solution,
        solve=lambda _residual, initial: initial,
        linear_solve=lambda _operator, rhs: rhs,
    )

    result = root(None, initial=False)

    assert result is False
