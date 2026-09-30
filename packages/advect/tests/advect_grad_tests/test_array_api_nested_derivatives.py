"""Nested Array API derivatives where traced values meet concrete provider arrays.

Differentiating a transform with respect to one of its tangents, primals or
cotangents traces that value, while the other operands and every value a rule
reads from them stay Array API arrays. A standard array rejects a traced
right-hand operand of ``*``, ``+``, ``-`` or ``@``, so each rule must keep
working when either side of an operation is the traced one.
"""

from __future__ import annotations

import math
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def _strict_namespace(value: Any) -> Any:
    return value.__array_namespace__()


@pytest.mark.parametrize("staged", [False, True], ids=["dynamic", "staged"])
def test_array_api_hvp_differentiates_through_a_traced_comparison(*, staged: bool) -> None:
    """Forward-over-reverse reaches the comparisons inside the abs transpose."""
    value = np.array([0.3, -1.2, 0.7])
    direction = np.array([1.0, 0.5, -0.25])

    def product(v: Any, t: Any) -> Any:
        loss = ad.hvp(lambda u: _strict_namespace(u).sum(_strict_namespace(u).abs(u) ** 3))
        return loss(v, vectors=t)[1]

    if staged:
        product = ad.stage(product, specs=(ad.ArraySpec(value.shape, "float64"),) * 2)
    actual = product(
        strict.asarray(value, dtype=strict.float64),
        strict.asarray(direction, dtype=strict.float64),
    )

    assert_allclose(np.asarray(actual), 6.0 * np.abs(value) * direction)


_VECTOR = strict.asarray([0.3, -1.2, 0.7])
_OTHER_VECTOR = strict.asarray([0.5, -1.0, 2.0])
_MATRIX = strict.asarray([[2.0, 0.3, -0.1], [0.4, 1.5, 0.2], [-0.3, 0.1, 1.2]])
_OTHER_MATRIX = strict.asarray([[1.0, -0.3, 0.2], [0.1, 0.8, -0.4], [0.5, 0.2, 1.1]])
_SYMMETRIC = strict.asarray([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.2]])
_COMPLEX_MATRIX = strict.asarray(
    [[2.0 + 0.1j, 0.3, -0.1j], [0.4, 1.5 - 0.2j, 0.2], [-0.3j, 0.1, 1.2]],
)
_COMPLEX_VECTOR = strict.asarray([0.3 + 0.1j, -1.2 + 0.5j, 0.7 - 0.2j])
# Power-of-two pivot ratios make each determinant exactly zero on any LAPACK.
_SINGULAR = strict.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.5, -1.0, 2.0]])
_RANK_ONE = strict.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
_SQUARE = strict.asarray([[0.3, -1.2, 0.7], [2.1, 0.4, -0.5], [1.5, -0.8, 0.9]])


def _linalg(value: Any) -> Any:
    return _strict_namespace(value).linalg


def _total(value: Any) -> Any:
    """Sum the real parts of every output leaf."""
    leaves = value if isinstance(value, tuple) else (value,)
    xp = _strict_namespace(leaves[0])
    sums = [
        xp.sum(xp.real(leaf) if xp.isdtype(leaf.dtype, "complex floating") else leaf)
        for leaf in leaves
    ]
    return sum(sums[1:], start=sums[0])


_TANGENT_CASES = [
    ("binary", lambda a, b: a * b + a / b, (_VECTOR, _OTHER_VECTOR)),
    ("multiply_constant", lambda v: v * strict.asarray([3.0, -5.0, 0.5]), (_VECTOR,)),
    ("scale", lambda v: v * 0.5, (_VECTOR,)),
    ("sin", lambda v: _strict_namespace(v).sin(v), (_VECTOR,)),
    ("sign_complex", lambda v: _strict_namespace(v).sign(v), (_COMPLEX_VECTOR,)),
    (
        "clip",
        lambda v, lo, hi: _strict_namespace(v).clip(v, lo, hi),
        (_VECTOR, _VECTOR - 0.5, _VECTOR + 0.1),
    ),
    ("prod", lambda v: _strict_namespace(v).prod(v), (_VECTOR,)),
    ("cumulative_prod", lambda v: _strict_namespace(v).cumulative_prod(v), (_VECTOR,)),
    ("var", lambda m: _strict_namespace(m).var(m, axis=0), (_MATRIX,)),
    ("vector_norm", lambda v: _linalg(v).vector_norm(v, ord=3), (_VECTOR,)),
    ("matrix_norm_fro", lambda m: _linalg(m).matrix_norm(m), (_MATRIX,)),
    ("matrix_norm_nuc", lambda m: _linalg(m).matrix_norm(m, ord="nuc"), (_MATRIX,)),
    ("matrix_norm_1", lambda m: _linalg(m).matrix_norm(m, ord=1), (_MATRIX,)),
    ("trace_dtype", lambda m: _linalg(m).trace(m, dtype=strict.float32), (_MATRIX,)),
    ("matmul", lambda a, b: a @ b, (_MATRIX, _OTHER_MATRIX)),
    ("vecdot", lambda a, b: _linalg(a).vecdot(a, b), (_VECTOR, _OTHER_VECTOR)),
    ("solve", lambda a, b: _linalg(a).solve(a, b), (_MATRIX, _VECTOR)),
    ("det", lambda m: _linalg(m).det(m), (_MATRIX,)),
    ("det_singular", lambda m: _linalg(m).det(m), (strict.stack([_SINGULAR, _MATRIX]),)),
    ("det_rank_one", lambda m: _linalg(m).det(m), (_RANK_ONE,)),
    ("slogdet", lambda m: _linalg(m).slogdet(m), (_COMPLEX_MATRIX,)),
    ("inv", lambda m: _linalg(m).inv(m), (_MATRIX,)),
    ("pinv", lambda m: _linalg(m).pinv(m), (_MATRIX,)),
    ("qr", lambda m: _linalg(m).qr(m), (_MATRIX,)),
    ("cholesky", lambda m: _linalg(m).cholesky(m), (_SYMMETRIC,)),
    ("eigh", lambda m: _linalg(m).eigh(m), (_SYMMETRIC,)),
    ("eigvalsh", lambda m: _linalg(m).eigvalsh(m), (_SYMMETRIC,)),
    ("svd", lambda m: _linalg(m).svd(m, full_matrices=False), (_MATRIX,)),
    ("svd_complex", lambda m: _linalg(m).svd(m, full_matrices=False), (_COMPLEX_MATRIX,)),
    ("svdvals", lambda m: _linalg(m).svdvals(m), (_MATRIX,)),
]


@pytest.mark.parametrize(
    ("function", "arguments", "index"),
    [
        pytest.param(function, arguments, index, id=f"{name}-{index}")
        for name, function, arguments in _TANGENT_CASES
        for index in range(len(arguments))
    ],
)
def test_array_api_jvp_differentiates_in_one_traced_tangent(
    function: Any,
    arguments: tuple[Any, ...],
    index: int,
) -> None:
    """A JVP is linear in each tangent; one tangent's gradient is a pullback.

    Differentiating with respect to one tangent traces it, while the primals,
    the partials evaluated at them and the other tangents stay Array API
    arrays, which reject a traced right-hand operand of ``*``, ``+``, ``-``
    or ``@``.
    """
    argnums = tuple(range(len(arguments)))
    ones = tuple(_strict_namespace(value).ones_like(value) for value in arguments)

    def pushed(tangent: Any) -> Any:
        tangents = (*ones[:index], tangent, *ones[index + 1 :])
        _output, output_tangent = ad.jvp(function, argnums=argnums)(*arguments, tangents=tangents)
        return _total(output_tangent)

    expected = ad.grad(lambda *values: _total(function(*values)), argnums=argnums)(*arguments)
    actual = ad.grad(pushed)(ones[index])
    assert_allclose(np.asarray(actual), np.asarray(expected[index]), rtol=1e-12, atol=1e-12)


def _binary(name: str) -> Any:
    return lambda a, b: getattr(_strict_namespace(a), name)(a, b)


def _svd_moduli(value: Any) -> Any:
    """Return a complex SVD's phase-invariant squared moduli and values.

    A phase convention for the singular vectors is a connection rather than
    a function of the matrix, so only phase-invariant functions of them have
    symmetric second derivatives.
    """
    xp = _strict_namespace(value)
    u, singular_values, vh = xp.linalg.svd(value, full_matrices=False)
    return xp.real(u * xp.conj(u)), singular_values, xp.real(vh * xp.conj(vh))


_POSITIVE_VECTOR = strict.asarray([0.3, 1.2, 0.7])
_WIDE_MATRIX = strict.asarray([[2.0, 0.3, -0.1, 0.5], [0.4, 1.5, 0.2, -0.7]])
_PRIMAL_CASES = [
    ("sign_complex", lambda v: _strict_namespace(v).sign(v), (_COMPLEX_VECTOR,)),
    ("divide", _binary("divide"), (_VECTOR, _OTHER_VECTOR)),
    ("remainder", _binary("remainder"), (_VECTOR, _OTHER_VECTOR)),
    ("pow", _binary("pow"), (_POSITIVE_VECTOR, _OTHER_VECTOR)),
    ("atan2", _binary("atan2"), (_VECTOR, _OTHER_VECTOR)),
    ("hypot", _binary("hypot"), (_VECTOR, _OTHER_VECTOR)),
    (
        "hypot_origin",
        _binary("hypot"),
        (strict.asarray([0.0, 0.3, 0.0]), strict.asarray([0.0, 0.0, -0.7])),
    ),
    ("copysign", _binary("copysign"), (_VECTOR, _OTHER_VECTOR)),
    ("logaddexp", _binary("logaddexp"), (_VECTOR, _OTHER_VECTOR)),
    ("maximum", _binary("maximum"), (_VECTOR, _OTHER_VECTOR)),
    ("minimum", _binary("minimum"), (_VECTOR, _OTHER_VECTOR)),
    ("cumulative_prod", lambda v: _strict_namespace(v).cumulative_prod(v), (_VECTOR,)),
    ("pinv", lambda m: _linalg(m).pinv(m), (_WIDE_MATRIX,)),
    ("svd", lambda m: _linalg(m).svd(m, full_matrices=False), (_MATRIX,)),
    ("svd_complex", _svd_moduli, (_COMPLEX_MATRIX,)),
]


@pytest.mark.parametrize(
    ("function", "arguments", "index"),
    [
        pytest.param(function, arguments, index, id=f"{name}-{index}")
        for name, function, arguments in _PRIMAL_CASES
        for index in range(len(arguments))
    ],
)
def test_array_api_jvp_differentiates_in_one_traced_primal(
    function: Any,
    arguments: tuple[Any, ...],
    index: int,
) -> None:
    """A JVP's derivative in one primal is a Hessian-vector product.

    Differentiating with respect to one primal traces it, while the other
    primals and every tangent stay Array API arrays, so a partial or a
    tangent can meet a traced operand on its right. Reverse and forward
    differentiation of the JVP must both agree with ``hvp``.
    """
    argnums = tuple(range(len(arguments)))
    ones = tuple(_strict_namespace(value).ones_like(value) for value in arguments)

    def directional(value: Any) -> Any:
        values = (*arguments[:index], value, *arguments[index + 1 :])
        _output, output_tangent = ad.jvp(function, argnums=argnums)(*values, tangents=ones)
        return _total(output_tangent)

    _value, products = ad.hvp(lambda *values: _total(function(*values)), argnums=argnums)(
        *arguments,
        vectors=ones,
    )
    expected = np.asarray(products[index])
    reverse = ad.grad(directional)(arguments[index])
    _output, forward = ad.jvp(directional)(arguments[index], tangents=ones[index])
    assert_allclose(np.asarray(reverse), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(forward), np.sum(np.real(expected)), rtol=1e-12, atol=1e-12)


_CONSTANT = strict.asarray([0.5, -1.0, 2.0])


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        pytest.param(
            lambda v: _strict_namespace(v).divide(_CONSTANT, v),
            lambda v: np.asarray(_CONSTANT) / v,
            id="divide",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).atan2(_CONSTANT, v),
            lambda v: np.arctan2(np.asarray(_CONSTANT), v),
            id="atan2",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).maximum(_CONSTANT, v) ** 2,
            lambda v: np.maximum(np.asarray(_CONSTANT), v) ** 2,
            id="maximum",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).pow(v, _CONSTANT),
            lambda v: v ** np.asarray(_CONSTANT),
            id="pow",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).matmul(_CONSTANT, v) ** 2,
            lambda v: (np.asarray(_CONSTANT) @ v) ** 2,
            id="matmul",
        ),
        pytest.param(
            lambda v: _strict_namespace(v).tensordot(_MATRIX, v, axes=1) ** 2,
            lambda v: np.tensordot(np.asarray(_MATRIX), v, axes=1) ** 2,
            id="tensordot",
        ),
    ],
)
def test_array_api_hvp_takes_a_constant_operand(function: Any, reference: Any) -> None:
    """A constant operand stays an Array API array beside a traced one."""
    value = strict.asarray([0.3, 1.2, 0.7])
    direction = strict.asarray([1.0, -0.5, 2.0])
    _value, product = ad.hvp(lambda v: _strict_namespace(v).sum(function(v)))(
        value,
        vectors=direction,
    )
    _value, expected = ad.hvp(lambda v: np.sum(reference(v)))(
        np.asarray(value),
        vectors=np.asarray(direction),
    )
    assert_allclose(np.asarray(product), expected, rtol=1e-12, atol=1e-12)


def _weights_like(leaf: Any) -> Any:
    """Return concrete, uneven real weights with a leaf's shape and dtype."""
    shape = tuple(int(extent) for extent in leaf.shape)
    values = np.cos(np.arange(math.prod(shape)) + 0.5).reshape(shape)
    return strict.asarray(values, dtype=leaf.dtype)


def _weighted_total(value: Any) -> Any:
    """Pair every output leaf with its weights under the real inner product."""
    leaves = ad.pytree.tree_leaves(value)
    xp = _strict_namespace(leaves[0])
    sums = [xp.sum(xp.real(leaf * _weights_like(leaf))) for leaf in leaves]
    return sum(sums[1:], start=sums[0])


def _traced_namespace(*values: Any) -> Any:
    """Return the namespace of a traced value among plain Array API arrays."""
    namespaces = [_strict_namespace(value) for value in values]
    return next((namespace for namespace in namespaces if namespace is not strict), strict)


_PULLBACK_CASES = [
    ("sin", lambda v: _strict_namespace(v).sin(v), (_VECTOR,)),
    # One primal node receives a concrete and a traced cotangent contribution.
    ("sin_plus_self", lambda v: _strict_namespace(v).sin(v) + v, (_VECTOR,)),
    (
        "remainder_by_self",
        lambda v: _strict_namespace(v).remainder(v, 0.5 * v + 2.0),
        (_VECTOR,),
    ),
    (
        "multiply",
        lambda a, b: _traced_namespace(a, b).multiply(a, b),
        (_VECTOR, _OTHER_VECTOR),
    ),
    ("abs_complex", lambda v: _strict_namespace(v).abs(v), (_COMPLEX_VECTOR,)),
    ("sign_complex", lambda v: _strict_namespace(v).sign(v), (_COMPLEX_VECTOR,)),
    (
        "matmul_vectors",
        lambda a, b: _traced_namespace(a, b).matmul(a, b),
        (_VECTOR, _OTHER_VECTOR),
    ),
    (
        "matmul_vector_matrix",
        lambda a, b: _traced_namespace(a, b).matmul(a, b),
        (_VECTOR, _MATRIX),
    ),
    (
        "vecdot",
        lambda a, b: _traced_namespace(a, b).linalg.vecdot(a, b),
        (_VECTOR, _OTHER_VECTOR),
    ),
    (
        "tensordot",
        lambda a, b: _traced_namespace(a, b).tensordot(a, b, axes=1),
        (_MATRIX, _OTHER_MATRIX),
    ),
    (
        "solve",
        lambda a, b: _traced_namespace(a, b).linalg.solve(a, b),
        (_MATRIX, _VECTOR),
    ),
    ("cholesky", lambda m: _linalg(m).cholesky(m), (_SYMMETRIC,)),
    ("eigh", lambda m: _linalg(m).eigh(m), (_SYMMETRIC,)),
    ("eigvalsh", lambda m: _linalg(m).eigvalsh(m), (_SYMMETRIC,)),
    ("pinv", lambda m: _linalg(m).pinv(m), (_WIDE_MATRIX,)),
    ("qr", lambda m: _linalg(m).qr(m), (_MATRIX,)),
    ("svd", lambda m: _linalg(m).svd(m, full_matrices=False), (_MATRIX,)),
    ("svdvals", lambda m: _linalg(m).svdvals(m), (_MATRIX,)),
    ("imag", lambda v: _strict_namespace(v).imag(v * (1.0 + 2.0j)), (_SQUARE,)),
    ("sort", lambda v: _strict_namespace(v).sort(v, axis=-1), (_SQUARE,)),
    ("trace", lambda m: _linalg(m).trace(m, offset=1), (_SQUARE,)),
    ("diagonal", lambda m: _linalg(m).diagonal(m, offset=-1), (_SQUARE,)),
]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [pytest.param(function, arguments, id=name) for name, function, arguments in _PULLBACK_CASES],
)
def test_array_api_vjp_differentiates_in_its_traced_cotangent(
    function: Any,
    arguments: tuple[Any, ...],
) -> None:
    """A pullback is linear in its cotangent, also where the cotangent is traced.

    Differentiating with respect to the cotangent traces it, while the primals
    and every value a rule reads from them stay Array API arrays. In reverse
    mode the derivative of the pulled total is the JVP along ones; in forward
    mode the pullback pushes its own cotangent forward.
    """
    argnums = tuple(range(len(arguments)))
    ones = tuple(_strict_namespace(value).ones_like(value) for value in arguments)
    output, pullback = ad.vjp(function, argnums=argnums)(*arguments)
    seed = ad.pytree.tree_map(lambda leaf: _strict_namespace(leaf).ones_like(leaf), output)
    weights = ad.pytree.tree_map(_weights_like, output)

    def pull(cotangent: Any) -> Any:
        _output, fresh = ad.vjp(function, argnums=argnums)(*arguments)
        return fresh(cotangent)

    actual = ad.grad(lambda cotangent: _total(pullback(cotangent)))(seed)
    _output, expected = ad.jvp(function, argnums=argnums)(*arguments, tangents=ones)
    pulled = pull(weights)
    traced_pulled, forward = ad.jvp(pull)(weights, tangents=weights)
    for actual_leaf, expected_leaf in zip(
        ad.pytree.tree_leaves(actual),
        ad.pytree.tree_leaves(expected),
        strict=True,
    ):
        assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf), rtol=1e-12, atol=1e-12)
    for leaves in zip(
        ad.pytree.tree_leaves(pulled),
        ad.pytree.tree_leaves(traced_pulled),
        ad.pytree.tree_leaves(forward),
        strict=True,
    ):
        for leaf in leaves[1:]:
            assert_allclose(np.asarray(leaf), np.asarray(leaves[0]), rtol=1e-12, atol=1e-12)


_WIDE_DECOMPOSITIONS = {
    "qr": lambda m: _linalg(m).qr(m),
    "svd": lambda m: _linalg(m).svd(m, full_matrices=False),
}


@pytest.mark.parametrize(
    ("name", "traced"),
    [("qr", 0), ("qr", 1), ("svd", 0), ("svd", 1), ("svd", 2)],
)
def test_array_api_wide_decomposition_vjp_differentiates_in_one_cotangent_leaf(
    name: str,
    traced: int,
) -> None:
    """Only one cotangent leaf is traced; the others stay concrete Array API arrays.

    The pullback is linear in that leaf, so the derivative of its total is the
    JVP's matching output leaf along ones.
    """
    function = _WIDE_DECOMPOSITIONS[name]
    output, pullback = ad.vjp(function)(_WIDE_MATRIX)
    seeds, treedef = ad.pytree.tree_flatten(output)

    def loss(leaf: Any) -> Any:
        leaves = [leaf if index == traced else seed for index, seed in enumerate(seeds)]
        return _total(pullback(ad.pytree.tree_unflatten(treedef, leaves)))

    actual = ad.grad(loss)(seeds[traced])
    _output, tangent = ad.jvp(function)(_WIDE_MATRIX, tangents=strict.ones_like(_WIDE_MATRIX))
    expected = ad.pytree.tree_leaves(tangent)[traced]
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("function", "arguments", "index"),
    [
        pytest.param(function, arguments, index, id=f"{name}-{index}")
        for name, function, arguments in _PULLBACK_CASES
        for index in range(len(arguments))
    ],
)
def test_array_api_vjp_with_a_constant_cotangent_differentiates_in_one_primal(
    function: Any,
    arguments: tuple[Any, ...],
    index: int,
) -> None:
    """A pullback of constant weights is a gradient, and its derivative an HVP.

    Differentiating with respect to one primal traces it, while the
    cotangent, the other primals and the values a rule reads from them stay
    Array API arrays.
    """

    def at(value: Any) -> tuple[Any, ...]:
        return (*arguments[:index], value, *arguments[index + 1 :])

    def pulled(value: Any) -> Any:
        output, pullback = ad.vjp(function, argnums=index)(*at(value))
        return _total(pullback(ad.pytree.tree_map(_weights_like, output)))

    value = arguments[index]
    _value, expected = ad.hvp(lambda v: _weighted_total(function(*at(v))))(
        value,
        vectors=_strict_namespace(value).ones_like(value),
    )
    actual = ad.grad(pulled)(value)
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12)
