"""Argument selection for the public autodiff transforms."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.autodiff.api.inputs import _get_signature


def test_get_signature_supports_unhashable_callable_instances() -> None:
    class _UnhashableCallable:
        def __call__(self, x: float, y: float = 1.0) -> float:
            return x + y

    type.__setattr__(_UnhashableCallable, "__hash__", None)
    callable_obj = _UnhashableCallable()
    sig = _get_signature(callable_obj)
    assert tuple(sig.parameters) == ("x", "y")


def test_get_signature_matches_a_staged_program_call() -> None:
    program = ad.stage(lambda x, y: x * y, specs=(ad.ArraySpec((), "float64"),) * 2)

    assert _get_signature(program) == inspect.signature(program)


def test_selections_resolve_against_the_callable_signature() -> None:
    named = ad.grad(lambda value, scale: value * scale, argnums=None, argnames=("scale",))
    variadic = ad.grad(lambda first, *rest: first * rest[0], argnums=-1)
    mixed = ad.grad(
        lambda value, *coefficients, scale: value * coefficients[0] * scale,
        argnums=1,
        argnames=("scale",),
    )

    # Products of these small binary fractions are exact.
    assert named(2.0, 3.0) == {"scale": 2.0}
    assert variadic(2.0, 3.0) == 2.0
    assert mixed(2.0, 3.0, scale=4.0) == (8.0, {"scale": 6.0})


def _objective(value: float, *extras: float, **options: float) -> float:
    return value + sum(extras) + sum(options.values())


# Names are checked against the signature when the transform is built;
# positions resolve against each call.
@pytest.mark.parametrize(
    ("argnums", "argnames", "at_construction", "error", "match"),
    [
        (None, ("missing",), True, ValueError, "Argument 'missing' not found"),
        (None, ("extras",), True, ValueError, "cannot select variadic parameter 'extras'"),
        (None, ("options",), True, ValueError, "cannot select variadic parameter 'options'"),
        (None, ("value", "value"), True, ValueError, "argnames contains duplicates"),
        (0, ("value",), False, ValueError, "selected by both argnums and argnames"),
        (2, None, False, IndexError, "index 2 is out of range"),
        ((0, -1), None, False, ValueError, "argnums contains duplicates"),
    ],
)
def test_invalid_selectors_fail_at_the_public_boundary(
    argnums: int | tuple[int, ...] | None,
    argnames: tuple[str, ...] | None,
    at_construction: bool,  # noqa: FBT001 - parametrized column
    error: type[Exception],
    match: str,
) -> None:
    def select() -> object:
        transform = ad.grad(_objective, argnums=argnums, argnames=argnames)
        return transform if at_construction else transform(1.0)

    with pytest.raises(error, match=match):
        select()


def test_selection_errors_name_the_selected_argument() -> None:
    with pytest.raises(IndexError, match="index 0 is out of range for 0"):
        ad.grad(lambda: 3.0)()
    with pytest.raises(ValueError, match="was not provided in the call"):
        ad.grad(lambda x, y=2.0: x * y, argnums=(), argnames=("y",))(3.0)
    with pytest.raises(ValueError, match=r"Cannot inspect signature.*max"):
        ad.grad(max, argnums=None, argnames=("value",))
    with pytest.raises(TypeError, match=r"arg0\['coefficient'\].*Python complex scalar"):
        ad.grad(max)({"coefficient": 1.0 + 2.0j}, 3.0)
    with pytest.raises(TypeError, match=r"tree\['complex'\].*Python complex scalar"):
        ad.grad(lambda tree: tree["real"] ** 2)({"real": 2.0, "complex": 1.0 + 2.0j})


def test_staged_selection_errors_fail_at_the_program_boundary() -> None:
    program = ad.stage(
        lambda value, *, scale: value * scale,
        specs=(ad.ArraySpec((2,), "float64"),),
        kw_specs={"scale": ad.ArraySpec((), "float64")},
    )
    labeled = ad.stage(
        lambda tree: tree["value"] * tree["value"],
        specs=({"value": ad.ArraySpec((), "float64"), "label": ad.StaticSpec("fixed")},),
    )

    with pytest.raises(ValueError, match="not provided as a keyword"):
        ad.jacobian(program, argnums=None, argnames=("scale",))(np.ones(2), np.array(2.0))
    with pytest.raises(ValueError, match="not present in the compiled signature"):
        ad.grad(program, argnums=None, argnames=("missing",))
    with pytest.raises(TypeError, match="cannot select static input leaves"):
        ad.grad(labeled)


def test_auxiliary_outputs_keep_the_callable_signature_for_selection() -> None:
    labels: list[str] = []

    def objective(weights: np.ndarray, *, scale: np.ndarray) -> tuple[object, np.ndarray]:
        labels.append(repr(weights))
        return np.sum(weights * scale), weights

    weights = np.array([1.0, 2.0])
    scale = np.array(3.0)
    with ad.debug():
        (weight_gradient, named), auxiliary = ad.grad(
            objective,
            argnums=0,
            argnames=("scale",),
            has_aux=True,
        )(weights, scale=scale)

    assert "[weights]" in labels[0]
    assert_allclose(weight_gradient, np.full(2, 3.0))
    assert_allclose(named["scale"], 3.0)
    assert_allclose(auxiliary, weights)

    program = ad.stage(
        objective,
        specs=(ad.ArraySpec((2,), "float64"),),
        kw_specs={"scale": ad.ArraySpec((), "float64")},
    )
    staged_named, staged_auxiliary = ad.grad(
        program,
        argnums=None,
        argnames=("scale",),
        has_aux=True,
    )(weights, scale=scale)

    assert_allclose(staged_named["scale"], 3.0)
    assert_allclose(staged_auxiliary, weights)


def test_nested_transforms_keep_unselected_arguments_as_outer_dependencies() -> None:
    inner = ad.grad(
        lambda value, coefficients, *, bias: value * (coefficients["scale"] + bias),
        argnums=0,
    )
    outer = ad.grad(
        lambda scale, bias: inner(2.0, {"scale": scale}, bias=bias),
        argnums=(0, 1),
    )
    named = ad.grad(lambda value, scale: value * scale, argnums=(), argnames=("scale",))

    assert outer(3.0, 4.0) == pytest.approx((1.0, 1.0))
    assert ad.grad(lambda value: named(value, 3.0)["scale"])(2.0) == pytest.approx(1.0)


def test_nested_named_keyword_selection_keeps_the_outer_trace_passive() -> None:
    inner = ad.grad(
        lambda value, *, scale: value * scale,
        argnums=(),
        argnames=("scale",),
    )
    outer = ad.grad(lambda scale: inner(2.0, scale=scale)["scale"])

    assert inner(2.0, scale=3.0) == {"scale": pytest.approx(2.0)}
    assert outer(3.0) == pytest.approx(0.0)
