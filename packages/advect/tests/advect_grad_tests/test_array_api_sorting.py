"""Portable sorting and search qualification."""

from __future__ import annotations

import pytest

import advect as ad

xp = pytest.importorskip("array_api_strict")


def _roundtrip(program: ad.StagedProgram) -> ad.StagedProgram:
    return ad.StagedProgram.from_dict(program.to_dict())


def _sorted_weighted_sum(value: object) -> object:
    namespace = value.__array_namespace__()  # type: ignore[attr-defined]
    weights = namespace.asarray([1.0, 2.0, 3.0], dtype=value.dtype)  # type: ignore[attr-defined]
    return namespace.sum(namespace.sort(value) * weights)


def test_sort_gradient_is_reordered_by_the_primal_permutation() -> None:
    value = xp.asarray([3.0, 1.0, 2.0], dtype=xp.float64)
    expected = xp.asarray([3.0, 1.0, 2.0], dtype=xp.float64)
    program = ad.stage(
        _sorted_weighted_sum,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    dynamic = ad.grad(_sorted_weighted_sum)(value)
    staged = ad.grad(program)(value)
    serialized = ad.grad(_roundtrip(program))(value)

    assert xp.all(xp.abs(dynamic - expected) < 1e-12)
    assert xp.all(xp.abs(staged - expected) < 1e-12)
    assert xp.all(xp.abs(serialized - expected) < 1e-12)


@pytest.mark.parametrize("operation", ["argsort", "searchsorted"])
def test_discrete_sorting_operations_roundtrip_but_do_not_differentiate(
    operation: str,
) -> None:
    value = xp.asarray([3.0, 1.0, 2.0], dtype=xp.float64)

    if operation == "argsort":

        def function(argument: object) -> object:
            namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
            return namespace.argsort(argument, stable=True)

        expected = xp.argsort(value, stable=True)
    else:
        queries = xp.asarray([0.5, 2.5, 4.0], dtype=xp.float64)

        def function(argument: object) -> object:
            namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
            return namespace.searchsorted(argument, queries)

        value = xp.sort(value)
        expected = xp.searchsorted(value, queries)

    program = ad.stage(
        function,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )

    assert xp.all(function(value) == expected)
    assert xp.all(program(value) == expected)
    assert xp.all(_roundtrip(program)(value) == expected)

    def loss(argument: object) -> object:
        namespace = argument.__array_namespace__()  # type: ignore[attr-defined]
        indices = function(argument)
        return namespace.sum(namespace.astype(indices, argument.dtype))  # type: ignore[attr-defined]

    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(loss)(value)

    loss_program = ad.stage(
        loss,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    restored_loss = _roundtrip(loss_program)
    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(loss_program)
    with pytest.raises(ad.NoVJPError, match="non-differentiable"):
        ad.grad(restored_loss)
