"""Strict derivative and lifetime assertions shared by the NumPy frontend tests."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any

import numpy as np

import advect as ad
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.core._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def assert_tree_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert equal pytree structure, then leaf shape, dtype, and values."""
    actual_leaves, actual_tree = tree_flatten(actual)
    expected_leaves, expected_tree = tree_flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=rtol, atol=atol, equal_nan=True
        )


def real_inner_product(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Return ``Re(sum(conj(left) * right))`` summed over paired leaves."""
    return sum(
        float(np.real(np.vdot(np.asarray(a), np.asarray(b))))
        for a, b in zip(left, right, strict=True)
    )


def seeded_like(value: Any, salt: str) -> np.ndarray[Any, Any]:
    """Return a deterministic nonuniform array with ``value``'s shape and dtype.

    All-ones directions cannot tell a Jacobian from its transpose, so every
    derivative check draws independent directions and cotangents instead.
    Non-inexact leaves carry no derivative and get zeros.
    """
    array = np.asarray(value)
    if array.dtype.kind not in "fc":
        return np.zeros(array.shape)
    generator = np.random.default_rng(zlib.crc32(salt.encode()))
    sample = generator.uniform(-1.0, 1.0, array.shape)
    if array.dtype.kind == "c":
        sample = sample + 1j * generator.uniform(-1.0, 1.0, array.shape)
    return sample.astype(array.dtype)


def assert_adjoint_identity(
    function: Callable[..., Any],
    primals: Sequence[Any],
    directions: Sequence[Any],
    tangent: Any,
    *,
    argnums: tuple[int, ...],
    salt: str = "",
) -> tuple[Any, Any, tuple[Any, ...]]:
    """Assert ``Re<w, J u> == Re<J^T w, u>`` for a seeded cotangent ``w``.

    Each input cotangent must also have its primal's shape and dtype. Return
    the VJP's primal, the cotangent seed, and the input cotangents.
    """
    tangent_leaves, tangent_tree = tree_flatten(tangent)
    seeds = [
        seeded_like(leaf, f"{salt}:cotangent:{index}") for index, leaf in enumerate(tangent_leaves)
    ]
    seed = tree_unflatten(tangent_tree, seeds)
    primal, pullback = ad.vjp(function, argnums=argnums)(*primals)
    try:
        cotangents = tuple(pullback(seed))
    finally:
        pullback.close()
    for cotangent, index in zip(cotangents, argnums, strict=True):
        assert np.shape(cotangent) == np.shape(primals[index])
        assert np.asarray(cotangent).dtype == np.asarray(primals[index]).dtype
    pairs = (*zip(seeds, tangent_leaves, strict=True), *zip(cotangents, directions, strict=True))
    forward = real_inner_product(seeds, tangent_leaves)
    reverse = real_inner_product(cotangents, directions)
    scale = sum(float(np.sum(np.abs(left) * np.abs(right))) for left, right in pairs)
    epsilon = max(
        (
            np.finfo(np.asarray(leaf).dtype).eps
            for pair in pairs
            for leaf in pair
            if np.asarray(leaf).dtype.kind in "fc"
        ),
        default=0.0,
    )
    assert abs(forward - reverse) <= 1e3 * epsilon * max(scale, 1.0), (forward, reverse)
    return primal, seed, cotangents


def assert_jvp_matches_central_difference(
    function: Callable[..., Any],
    primals: Sequence[Any],
    directions: Sequence[Any],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
    step: float = 1e-6,
    adjoint: bool = True,
) -> tuple[Any, Any]:
    """Check a JVP against NumPy, central differences, and its own VJP.

    The primal must match ``function(*primals)`` in structure, shape, dtype and
    value (``rtol=1e-7`` with the caller's ``atol``); the tangent must have the
    primal's structure and shapes, the dtype of every inexact primal leaf, and
    the central difference's values within ``rtol``/``atol``. Unless
    ``adjoint=False``, the VJP must satisfy the real adjoint identity.
    """
    argnums = tuple(range(len(primals)))
    primal, tangent = ad.jvp(function, argnums=argnums)(*primals, tangents=tuple(directions))
    assert_tree_close(primal, function(*primals), atol=atol)

    def shifted(sign: float) -> Any:
        return function(
            *(
                np.asarray(value + sign * step * np.asarray(direction))
                for value, direction in zip(primals, directions, strict=True)
            )
        )

    primal_leaves, primal_tree = tree_flatten(primal)
    tangent_leaves, tangent_tree = tree_flatten(tangent)
    plus_leaves, plus_tree = tree_flatten(shifted(1.0))
    minus_leaves, minus_tree = tree_flatten(shifted(-1.0))
    assert tangent_tree == primal_tree == plus_tree == minus_tree
    for primal_leaf, tangent_leaf, upper, lower in zip(
        primal_leaves, tangent_leaves, plus_leaves, minus_leaves, strict=True
    ):
        primal_array = np.asarray(primal_leaf)
        tangent_array = np.asarray(tangent_leaf)
        assert tangent_array.shape == primal_array.shape
        if primal_array.dtype.kind in "fc":
            assert tangent_array.dtype == primal_array.dtype
        np.testing.assert_allclose(
            tangent_array,
            (np.asarray(upper) - np.asarray(lower)) / (2 * step),
            rtol=rtol,
            atol=atol,
        )
    if adjoint:
        assert_adjoint_identity(function, primals, directions, tangent, argnums=argnums)
    return primal, tangent


def assert_spellings_agree(
    function: Callable[..., Any],
    reference: Callable[..., Any],
    values: Sequence[Any],
    *,
    argnums: tuple[int, ...],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Any:
    """Assert two spellings of one array computation trace to the same derivatives.

    Both run with the same seeded directions and cotangent: their primals must
    be equal in shape, dtype and value, and their JVP tangents and VJP
    cotangents equal within ``rtol``/``atol``. Return the traced primal.
    """
    directions = tuple(seeded_like(values[index], f"direction:{index}") for index in argnums)

    def derivatives(call: Callable[..., Any]) -> tuple[Any, Any, tuple[Any, ...]]:
        primal, tangent = ad.jvp(call, argnums=argnums)(*values, tangents=directions)
        _, pullback = ad.vjp(call, argnums=argnums)(*values)
        try:
            return primal, tangent, tuple(pullback(seeded_like(primal, "cotangent")))
        finally:
            pullback.close()

    (primal, *actual), (expected_primal, *expected) = map(derivatives, (function, reference))
    assert_tree_close(primal, expected_primal, rtol=0.0)
    assert_tree_close(actual, expected, rtol=rtol, atol=atol)
    return primal


def assert_staged_round_trip(
    function: Callable[..., Any],
    *values: Any,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> ad.StagedProgram:
    """Stage ``function`` and match eager NumPy before and after serialization.

    Specs alone select Advect's latest Array API revision, so the program is
    compiled at the newest revision the installed NumPy serves.
    """
    program = ad.stage(
        function,
        specs=tuple(ad.ArraySpec(np.shape(value), np.asarray(value).dtype) for value in values),
        array_api_version=min(np.__array_api_version__, LATEST_ARRAY_API_VERSION),
    )
    expected = function(*values)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        assert_tree_close(staged(*values), expected, rtol=rtol, atol=atol)
    return program


__all__ = [
    "assert_adjoint_identity",
    "assert_jvp_matches_central_difference",
    "assert_spellings_agree",
    "assert_staged_round_trip",
    "assert_tree_close",
    "real_inner_product",
    "seeded_like",
]
