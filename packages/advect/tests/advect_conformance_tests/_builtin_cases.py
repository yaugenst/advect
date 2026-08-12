"""Conformance cases for Advect's built-in primitives.

This table is the single place a built-in operation declares how it should be
exercised. To cover a new primitive, append one :class:`InvocationCase`;
:mod:`test_registry_coverage` fails until you do, and
:mod:`test_builtin_conformance` runs the full law battery once you have.

An operation can and often should have more than one invocation. Frontends,
positional/keyword signatures, and materially different static attributes are
separate contracts even when they resolve to the same registry rule.

Choosing a domain is the only real judgement call. The rule of thumb: pick the
domain that makes the primitive *smooth and well conditioned* on every value it
can draw. If no such domain exists, the primitive is genuinely kinked or
genuinely flat somewhere -- say so by narrowing ``laws`` and recording
``reason``, rather than by loosening tolerances until the suite goes quiet.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from advect_conformance_tests._harness import (
    DEFAULT_LAWS,
    Argument,
    ClipRegions,
    Distinct,
    Frontend,
    Increasing,
    InputVariant,
    Interior,
    InvocationCase,
    Law,
    Nonzero,
    NumericalReference,
    Positive,
    Real,
    SeparatedFrom,
    SpanningGrid,
    StableEigensystem,
    SymmetricPositiveDefinite,
    Tolerance,
    Unit,
    WellConditioned,
)

_VECTOR = (6,)
_MATRIX = (4, 4)

#: Decompositions and inverses lose digits in proportion to the condition
#: number, which the domains bound but cannot eliminate. This factor is kept
#: narrow enough that a percent-level rule defect cannot pass.
_LINALG_TOLERANCE = Tolerance().scaled(30)
_CAST_TOLERANCE = Tolerance(
    adjoint_rtol=1e-6,
    adjoint_atol=1e-6,
    finite_difference_rtol=1e-3,
    finite_difference_atol=1e-4,
    finite_difference_step=1e-3,
)
_FLOAT32_TOLERANCE = Tolerance(
    primal_rtol=1e-6,
    primal_atol=1e-7,
    adjoint_rtol=1e-6,
    adjoint_atol=1e-6,
    finite_difference_rtol=1e-3,
    finite_difference_atol=1e-4,
    finite_difference_step=1e-3,
)
_LINALG_LOW_PRECISION_TOLERANCE = Tolerance(
    primal_rtol=1e-5,
    primal_atol=1e-6,
    adjoint_rtol=3e-5,
    adjoint_atol=3e-5,
    finite_difference_rtol=5e-3,
    finite_difference_atol=5e-4,
    finite_difference_step=1e-3,
)

_POINTWISE_REAL_VARIANTS = (
    InputVariant("scalar-float64", shapes={"x": ()}, dtypes={"x": "float64"}),
    InputVariant(
        "vector-float32",
        shapes={"x": (5,)},
        dtypes={"x": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
        numerical_reference=NumericalReference.CENTRAL,
    ),
    InputVariant("matrix-float64", shapes={"x": (2, 3)}, dtypes={"x": "float64"}),
    InputVariant(
        "tensor-float32",
        shapes={"x": (2, 1, 3)},
        dtypes={"x": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
        numerical_reference=NumericalReference.CENTRAL,
    ),
)
_POINTWISE_COMPLEX_VARIANTS = (
    InputVariant(
        "vector-complex64",
        shapes={"x": (5,)},
        dtypes={"x": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "matrix-complex128",
        shapes={"x": (2, 3)},
        dtypes={"x": "complex128"},
    ),
)
_POINTWISE_BINARY_VARIANTS = (
    InputVariant(
        "scalar-float64",
        shapes={"a": (), "b": ()},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "vector-float32",
        shapes={"a": (5,), "b": (5,)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "matrix-float64",
        shapes={"a": (2, 3), "b": (2, 3)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "broadcast-float32",
        shapes={"a": (2, 1, 3), "b": (1, 4, 1)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
)
_MATCHED_BINARY_VARIANTS = _POINTWISE_BINARY_VARIANTS[:-1]
_CONCATENATE_VARIANTS = (
    InputVariant(
        "vector-float32",
        shapes={"a": (3,), "b": (5,)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "matrix-float64",
        shapes={"a": (2, 3), "b": (4, 3)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "tensor-float32",
        shapes={"a": (2, 1, 3), "b": (3, 1, 3)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
)
_POINTWISE_COMPLEX_BINARY_VARIANTS = (
    InputVariant(
        "vector-complex64",
        shapes={"a": (5,), "b": (5,)},
        dtypes={"a": "complex64", "b": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "broadcast-complex128",
        shapes={"a": (2, 1, 3), "b": (1, 4, 1)},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
)
# The nonzero division domain permits denominator magnitudes down to 0.25.
# Squaring that denominator amplifies ordinary complex64 rule roundoff; the
# high-precision pairing oracle still keeps percent-scale formula defects far
# outside this operation-specific bound.
_COMPLEX64_DIVIDE_TOLERANCE = replace(
    _FLOAT32_TOLERANCE,
    adjoint_rtol=5e-6,
    adjoint_atol=5e-6,
)
_COMPLEX_DIVIDE_VARIANTS = (
    replace(
        _POINTWISE_COMPLEX_BINARY_VARIANTS[0],
        tolerance=_COMPLEX64_DIVIDE_TOLERANCE,
    ),
    _POINTWISE_COMPLEX_BINARY_VARIANTS[1],
)
_SQUARE_MATRIX_VARIANTS = (
    InputVariant("2x2-float64", shapes={"a": (2, 2)}, dtypes={"a": "float64"}),
    InputVariant(
        "3x3-float32",
        shapes={"a": (3, 3)},
        dtypes={"a": "float32"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant(
        "batch-2x2-float64",
        shapes={"a": (2, 2, 2)},
        dtypes={"a": "float64"},
    ),
    InputVariant(
        "2x2-complex64",
        shapes={"a": (2, 2)},
        dtypes={"a": "complex64"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant(
        "3x3-complex128",
        shapes={"a": (3, 3)},
        dtypes={"a": "complex128"},
    ),
)
_REAL_EIGEN_MATRIX_VARIANTS = _SQUARE_MATRIX_VARIANTS[:3]
_COMPLEX_EIGEN_MATRIX_VARIANTS = (
    *_SQUARE_MATRIX_VARIANTS[3:],
    InputVariant(
        "batch-2x2-complex128",
        shapes={"a": (2, 2, 2)},
        dtypes={"a": "complex128"},
    ),
)
_REDUCTION_VARIANTS = (
    InputVariant("vector-float64", shapes={"x": (5,)}, dtypes={"x": "float64"}),
    InputVariant(
        "matrix-float32",
        shapes={"x": (2, 3)},
        dtypes={"x": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
        numerical_reference=NumericalReference.CENTRAL,
    ),
    InputVariant("tensor-float64", shapes={"x": (2, 1, 3)}, dtypes={"x": "float64"}),
)
_COMPLEX_REDUCTION_VARIANTS = (
    InputVariant(
        "matrix-complex64",
        shapes={"x": (2, 3)},
        dtypes={"x": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "tensor-complex128",
        shapes={"x": (2, 1, 3)},
        dtypes={"x": "complex128"},
    ),
)
# A six-factor complex64 product can lose several ulps when its scalar JVP
# reduction is associated in reverse as a vector cotangent. Keep the observed
# bound on product reductions; the high-precision variant retains the strict
# default gate and percent-scale formula defects remain far outside this one.
_PRODUCT_COMPLEX64_TOLERANCE = replace(
    _FLOAT32_TOLERANCE,
    adjoint_rtol=1e-5,
    adjoint_atol=1e-5,
)
_PRODUCT_COMPLEX_REDUCTION_VARIANTS = (
    replace(_COMPLEX_REDUCTION_VARIANTS[0], tolerance=_PRODUCT_COMPLEX64_TOLERANCE),
    _COMPLEX_REDUCTION_VARIANTS[1],
)
# A six-factor float32 product can accumulate a few ulps of association
# difference between its scalar JVP pairing and elementwise pullback pairing.
# Keep that observed bound local to product reductions; float64 retains the
# strict default gate.
_PRODUCT_FLOAT32_TOLERANCE = replace(
    _FLOAT32_TOLERANCE,
    adjoint_rtol=2e-6,
    adjoint_atol=2e-6,
)
_PRODUCT_REDUCTION_VARIANTS = (
    _REDUCTION_VARIANTS[0],
    replace(_REDUCTION_VARIANTS[1], tolerance=_PRODUCT_FLOAT32_TOLERANCE),
    _REDUCTION_VARIANTS[2],
)
_FIXED_REAL_DTYPE_VARIANTS = (
    InputVariant("float64", dtypes={"x": "float64"}),
    InputVariant(
        "float32",
        dtypes={"x": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
)
# Six float32 scan products can accumulate a one-ulp difference between the
# forward pairing and reverse association while still satisfying the adjoint
# law. Keep that observed bound local to cumprod instead of weakening every
# float32 operation.
_CUMPROD_FLOAT32_TOLERANCE = replace(
    _FLOAT32_TOLERANCE,
    adjoint_rtol=2e-6,
    adjoint_atol=2e-6,
)
_CUMPROD_REDUCTION_VARIANTS = (
    _REDUCTION_VARIANTS[0],
    replace(_REDUCTION_VARIANTS[1], tolerance=_CUMPROD_FLOAT32_TOLERANCE),
    _REDUCTION_VARIANTS[2],
)
_CUMPROD_FIXED_REAL_DTYPE_VARIANTS = (
    _FIXED_REAL_DTYPE_VARIANTS[0],
    replace(_FIXED_REAL_DTYPE_VARIANTS[1], tolerance=_CUMPROD_FLOAT32_TOLERANCE),
)
_FIXED_NUMERIC_DTYPE_VARIANTS = (
    *_FIXED_REAL_DTYPE_VARIANTS,
    InputVariant(
        "complex64",
        dtypes={"x": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant("complex128", dtypes={"x": "complex128"}),
)
_MATMUL_VARIANTS = (
    InputVariant(
        "vector-vector-float64",
        shapes={"a": (3,), "b": (3,)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "matrix-vector-float32",
        shapes={"a": (2, 3), "b": (3,)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "vector-matrix-float64",
        shapes={"a": (3,), "b": (3, 2)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "matrix-matrix-complex64",
        shapes={"a": (2, 3), "b": (3, 4)},
        dtypes={"a": "complex64", "b": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "batched-broadcast-complex128",
        shapes={"a": (2, 1, 3, 4), "b": (1, 5, 4, 2)},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
    InputVariant(
        "batched-matrix-vector-complex128",
        shapes={"a": (2, 2, 3), "b": (3,)},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
)
_CONTRACTION_VARIANTS = (
    InputVariant(
        "vector-float64",
        shapes={"a": _VECTOR, "b": _VECTOR},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "scalar-complex128",
        shapes={"a": (), "b": ()},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
)
_SOLVE_VARIANTS = (
    InputVariant(
        "2x2-vector-float64",
        shapes={"a": (2, 2), "b": (2,)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "3x3-matrix-float32",
        shapes={"a": (3, 3), "b": (3, 2)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant(
        "batch-shared-2x2-matrix-float64",
        shapes={"a": (2, 2, 2), "b": (2, 2)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "batch-2x2-matrix-float64",
        shapes={"a": (2, 2, 2), "b": (2, 2, 3)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "2x2-vector-complex64",
        shapes={"a": (2, 2), "b": (2,)},
        dtypes={"a": "complex64", "b": "complex64"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant(
        "3x3-matrix-complex128",
        shapes={"a": (3, 3), "b": (3, 2)},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
)
_RECTANGULAR_MATRIX_VARIANTS = (
    InputVariant("tall-float64", shapes={"a": (4, 2)}, dtypes={"a": "float64"}),
    InputVariant(
        "wide-float32",
        shapes={"a": (2, 4)},
        dtypes={"a": "float32"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant(
        "tall-complex64",
        shapes={"a": (4, 2)},
        dtypes={"a": "complex64"},
        tolerance=_LINALG_LOW_PRECISION_TOLERANCE,
    ),
    InputVariant("wide-complex128", shapes={"a": (2, 4)}, dtypes={"a": "complex128"}),
)
_SIGNAL_VARIANTS = (
    InputVariant(
        "equal-float64",
        shapes={"a": (5,), "b": (5,)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "left-long-float32",
        shapes={"a": (7,), "b": (4,)},
        dtypes={"a": "float32", "b": "float32"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
    InputVariant(
        "right-long-float64",
        shapes={"a": (4,), "b": (7,)},
        dtypes={"a": "float64", "b": "float64"},
    ),
    InputVariant(
        "left-long-complex128",
        shapes={"a": (6,), "b": (3,)},
        dtypes={"a": "complex128", "b": "complex128"},
    ),
    InputVariant(
        "right-long-complex64",
        shapes={"a": (3,), "b": (6,)},
        dtypes={"a": "complex64", "b": "complex64"},
        tolerance=_FLOAT32_TOLERANCE,
    ),
)

#: For primitives whose derivative is identically zero by construction. They
#: still owe every other guarantee -- the forward value, a zero that transposes
#: to a matching zero, input immutability, and exact result metadata.
_FLAT = frozenset(
    {
        Law.PRIMAL,
        Law.FINITE_DIFFERENCE,
        Law.ADJOINT,
        Law.STRUCTURE,
        Law.NO_INPUT_MUTATION,
        Law.DTYPE,
    },
)

#: Values strictly inside one unit cell, so rounding primitives never sample
#: within a finite-difference step of a jump.
_BETWEEN_INTEGERS = Nonzero(margin=0.15, high=0.3)

#: ``d(a mod b)/db`` is ``-floor(a/b)``, which vanishes whenever the dividend is
#: smaller than the divisor. These bounds keep every possible broadcast pair
#: inside 2.28 < a / b < 2.62: both arguments carry derivative, and no sample is
#: near a wrap-around discontinuity.
_MODULO_DIVIDEND = Positive(low=3.2, high=3.4)
_MODULO_DIVISOR = Positive(low=1.3, high=1.4)


def _unary(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    """Declare a single-argument primitive over a vector."""
    return InvocationCase(
        op=op,
        call=call,
        arguments=(Argument("x", domain, shape=_VECTOR),),
        **kwargs,
    )


def _real_unary(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    kwargs.setdefault("variants", _POINTWISE_REAL_VARIANTS)
    return _unary(op, call, domain, **kwargs)


def _complex_unary(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    kwargs.setdefault("variants", _POINTWISE_COMPLEX_VARIANTS)
    kwargs.setdefault("reason", "complex64 and complex128 exercise the real-linear rule")
    return InvocationCase(
        op=op,
        call=call,
        arguments=(Argument("x", domain, shape=_VECTOR, dtype="complex128"),),
        **kwargs,
    )


def _analytic_unary(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    """Declare a real-domain holomorphic call with a strict complex-step oracle."""
    kwargs.setdefault("variants", _POINTWISE_REAL_VARIANTS)
    return _unary(
        op,
        call,
        domain,
        numerical_reference=NumericalReference.COMPLEX_STEP,
        **kwargs,
    )


def _binary(op: str, call: Any, left: Any, right: Any, **kwargs: Any) -> InvocationCase:
    """Declare a two-argument primitive with independent domains."""
    return InvocationCase(
        op=op,
        call=call,
        arguments=(
            Argument("a", left, shape=_VECTOR),
            Argument("b", right, shape=_VECTOR),
        ),
        **kwargs,
    )


def _real_binary(
    op: str,
    call: Any,
    left: Any,
    right: Any,
    *,
    broadcast: bool = True,
    **kwargs: Any,
) -> InvocationCase:
    kwargs.setdefault(
        "variants",
        _POINTWISE_BINARY_VARIANTS if broadcast else _MATCHED_BINARY_VARIANTS,
    )
    return _binary(op, call, left, right, **kwargs)


def _complex_binary(
    op: str,
    call: Any,
    left: Any,
    right: Any,
    *,
    variants: tuple[InputVariant, ...] = _POINTWISE_COMPLEX_BINARY_VARIANTS,
) -> InvocationCase:
    return InvocationCase(
        op=op,
        call=call,
        arguments=(
            Argument("a", left, shape=_VECTOR, dtype="complex128"),
            Argument("b", right, shape=_VECTOR, dtype="complex128"),
        ),
        variants=variants,
        reason="complex64 and complex128 cover matched and broadcast real-adjoint rules",
    )


def _reduction(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    kwargs.setdefault("variants", _REDUCTION_VARIANTS)
    return _unary(op, call, domain, **kwargs)


def _complex_reduction(op: str, call: Any, domain: Any) -> InvocationCase:
    return InvocationCase(
        op=op,
        call=call,
        arguments=(Argument("x", domain, shape=(2, 3), dtype="complex128"),),
        variants=_COMPLEX_REDUCTION_VARIANTS,
        reason="complex64 and complex128 cover multidimensional reduction rules",
    )


def _matrix(op: str, call: Any, domain: Any, **kwargs: Any) -> InvocationCase:
    kwargs.setdefault("variants", _SQUARE_MATRIX_VARIANTS)
    return InvocationCase(
        op=op,
        call=call,
        arguments=(Argument("a", domain, shape=_MATRIX),),
        **kwargs,
    )


def _xp(value: Any) -> Any:
    return value.__array_namespace__()


def _xp_call(value: Any, path: str, *arguments: Any, **kwargs: Any) -> Any:
    target = _xp(value)
    for component in path.split("."):
        target = getattr(target, component)
    return target(*arguments, **kwargs)


# --- elementwise, unbounded domain -------------------------------------------

_ELEMENTWISE_REAL: tuple[InvocationCase, ...] = (
    _analytic_unary(
        "array.sin",
        np.sin,
        Real(),
        laws=DEFAULT_LAWS | {Law.SECOND_ORDER},
        reason="smooth elementwise rule supports nested differentiation",
    ),
    _analytic_unary("array.cos", np.cos, Real()),
    _analytic_unary("array.tan", np.tan, Unit(margin=0.5)),
    _analytic_unary("array.sinh", np.sinh, Real()),
    _analytic_unary("array.cosh", np.cosh, Real()),
    _analytic_unary("array.tanh", np.tanh, Real()),
    _analytic_unary("array.exp", np.exp, Real()),
    _analytic_unary("array.expm1", np.expm1, Real()),
    _analytic_unary("array.arctan", np.arctan, Real()),
    _analytic_unary("array.arcsinh", np.arcsinh, Real()),
    _analytic_unary("array.negative", np.negative, Real()),
    _analytic_unary("array.positive", np.positive, Real()),
    _analytic_unary("array.square", np.square, Real()),
    _real_unary("array.conjugate", np.conjugate, Real()),
    _real_unary("array.real", np.real, Real()),
    _analytic_unary("array_ext.sinc", np.sinc, Nonzero()),
    _real_unary("array_ext.cbrt", np.cbrt, Nonzero()),
    _analytic_unary("array_ext.exp2", np.exp2, Real()),
    _real_unary("array_ext.degrees", np.degrees, Real()),
    _real_unary("array_ext.radians", np.radians, Real()),
    _real_unary("array_ext.deg2rad", np.deg2rad, Real()),
    _real_unary("array_ext.rad2deg", np.rad2deg, Real()),
)

_ANGLE = InvocationCase(
    op="array_ext.angle",
    call=lambda x: np.angle(x + 4.0),
    arguments=(
        Argument(
            "x",
            Real(),
            shape=_VECTOR,
            dtype="complex128",
        ),
    ),
    variants=_POINTWISE_COMPLEX_VARIANTS,
    reason="the positive real shift stays away from angle's negative-axis branch cut",
)

_COMPLEX_SIGN = InvocationCase(
    op="array.sign",
    call=np.sign,
    arguments=(Argument("x", Nonzero(), shape=_VECTOR, dtype="complex128"),),
    variants=_POINTWISE_COMPLEX_VARIANTS,
    reason="complex sign is a smooth real-linear unit-phase map away from zero",
)

_ELEMENTWISE_COMPLEX: tuple[InvocationCase, ...] = (
    _complex_unary("array.sin", np.sin, Real()),
    _complex_unary("array.cos", np.cos, Real()),
    _complex_unary("array.tan", np.tan, Real(scale=0.25)),
    _complex_unary("array.sinh", np.sinh, Real()),
    _complex_unary("array.cosh", np.cosh, Real()),
    _complex_unary("array.tanh", np.tanh, Real(scale=0.25)),
    _complex_unary("array.exp", np.exp, Real()),
    _complex_unary("array.expm1", np.expm1, Real()),
    _complex_unary("array.arctan", np.arctan, Real(scale=0.25)),
    _complex_unary("array.arcsinh", np.arcsinh, Real(scale=0.25)),
    _complex_unary("array.negative", np.negative, Real()),
    _complex_unary("array.positive", np.positive, Real()),
    _complex_unary("array.square", np.square, Real()),
    _complex_unary("array.conjugate", np.conjugate, Real()),
    _complex_unary("array.real", np.real, Real()),
    _complex_unary(
        "array.imag",
        lambda x: _xp_call(x, "imag", x),
        Real(),
        frontend=Frontend.ARRAY_API,
    ),
    _unary(
        "array.imag",
        np.imag,
        Real(),
        reason="real inputs exercise the identically zero imaginary-part rule",
    ),
    _complex_unary("array.log", lambda x: np.log(x + 4.0), Real()),
    _complex_unary("array.log2", lambda x: np.log2(x + 4.0), Real()),
    _complex_unary("array.log10", lambda x: np.log10(x + 4.0), Real()),
    _complex_unary("array.log1p", lambda x: np.log1p(x + 3.0), Real()),
    _complex_unary("array.sqrt", lambda x: np.sqrt(x + 4.0), Real()),
    _complex_unary("array.reciprocal", np.reciprocal, Nonzero()),
    _complex_unary("array.arcsin", np.arcsin, Real(scale=0.25)),
    _complex_unary("array.arccos", np.arccos, Real(scale=0.25)),
    _complex_unary("array.arctanh", np.arctanh, Real(scale=0.25)),
    _complex_unary("array.absolute", np.absolute, Nonzero()),
)

# --- elementwise, restricted domain ------------------------------------------

_ELEMENTWISE_RESTRICTED: tuple[InvocationCase, ...] = (
    _analytic_unary("array.log", np.log, Positive()),
    _analytic_unary("array.log2", np.log2, Positive()),
    _analytic_unary("array.log10", np.log10, Positive()),
    _analytic_unary("array.log1p", np.log1p, Positive()),
    _analytic_unary("array.sqrt", np.sqrt, Positive()),
    _analytic_unary("array.reciprocal", np.reciprocal, Nonzero()),
    _analytic_unary("array.arcsin", np.arcsin, Unit()),
    _analytic_unary("array.arccos", np.arccos, Unit()),
    _analytic_unary("array.arctanh", np.arctanh, Unit()),
    _analytic_unary("array.arccosh", np.arccosh, Positive(low=1.5, high=5.0)),
    # ``abs`` and ``fabs`` are kinked at the origin; the domain keeps every
    # sample away from it so the derivative laws stay meaningful.
    _real_unary("array.absolute", np.absolute, Nonzero()),
    _real_unary("array_ext.fabs", np.fabs, Nonzero()),
)

# --- elementwise with an identically zero derivative -------------------------

_FLAT_ELEMENTWISE: tuple[InvocationCase, ...] = (
    _real_unary("array.ceil", np.ceil, _BETWEEN_INTEGERS, laws=_FLAT, reason="piecewise constant"),
    _real_unary(
        "array.floor",
        np.floor,
        _BETWEEN_INTEGERS,
        laws=_FLAT,
        reason="piecewise constant",
    ),
    _real_unary("array.rint", np.rint, _BETWEEN_INTEGERS, laws=_FLAT, reason="piecewise constant"),
    _real_unary(
        "array.trunc",
        np.trunc,
        _BETWEEN_INTEGERS,
        laws=_FLAT,
        reason="piecewise constant",
    ),
    _real_unary("array.sign", np.sign, Nonzero(), laws=_FLAT, reason="piecewise constant"),
    _real_unary(
        "array_ext.spacing",
        np.spacing,
        Positive(low=0.3, high=0.45),
        laws=_FLAT,
        reason="piecewise constant within one floating-point binade",
    ),
)

# --- binary elementwise ------------------------------------------------------

_BINARY: tuple[InvocationCase, ...] = (
    _real_binary("array.add", np.add, Real(), Real()),
    _real_binary("array.subtract", np.subtract, Real(), Real()),
    _real_binary("array.multiply", np.multiply, Real(), Real()),
    _real_binary("array.divide", np.divide, Real(), Nonzero()),
    _real_binary("array.power", np.power, Positive(), Real(scale=0.5)),
    _real_binary("array_ext.float_power", np.float_power, Positive(), Real(scale=0.5)),
    _real_binary("array.arctan2", np.arctan2, Nonzero(), Nonzero()),
    _real_binary("array.hypot", np.hypot, Nonzero(), Nonzero()),
    _real_binary("array.nextafter", np.nextafter, Real(), Real()),
    _real_unary(
        "array_ext.heaviside",
        lambda x: np.heaviside(np.zeros_like(x), x),
        Real(),
    ),
    _real_unary(
        "array_ext.heaviside",
        lambda x: np.heaviside(x, 0.5),
        Nonzero(),
        laws=_FLAT,
        reason="piecewise constant away from the jump",
    ),
    _real_binary("array.logaddexp", np.logaddexp, Real(), Real()),
    _real_binary("array_ext.logaddexp2", np.logaddexp2, Real(), Real()),
    _real_unary(
        "array_ext.ldexp",
        lambda x: np.ldexp(x, np.asarray(3, dtype=np.int32)),
        Real(),
    ),
    InvocationCase(
        op="array.copysign",
        call=np.copysign,
        arguments=(
            Argument("a", Nonzero(), shape=_VECTOR),
            Argument("b", Nonzero(), shape=_VECTOR, differentiable=False),
        ),
        variants=_MATCHED_BINARY_VARIANTS,
    ),
    # Selection primitives are kinked exactly where the arguments tie, so the
    # domains make a tie structurally impossible rather than merely unlikely.
    _real_binary(
        "array.maximum",
        np.maximum,
        Real(),
        SeparatedFrom("a"),
        broadcast=False,
    ),
    _real_binary(
        "array.minimum",
        np.minimum,
        Real(),
        SeparatedFrom("a"),
        broadcast=False,
    ),
    _real_binary(
        "array_ext.fmax",
        np.fmax,
        Real(),
        SeparatedFrom("a"),
        broadcast=False,
    ),
    _real_binary(
        "array_ext.fmin",
        np.fmin,
        Real(),
        SeparatedFrom("a"),
        broadcast=False,
    ),
    _real_binary("array.remainder", np.remainder, _MODULO_DIVIDEND, _MODULO_DIVISOR),
    _real_binary("array_ext.fmod", np.fmod, _MODULO_DIVIDEND, _MODULO_DIVISOR),
    _real_binary(
        "array.floor_divide",
        np.floor_divide,
        _MODULO_DIVIDEND,
        _MODULO_DIVISOR,
        laws=_FLAT,
        reason="piecewise constant away from quotient boundaries",
    ),
)

_BINARY_COMPLEX: tuple[InvocationCase, ...] = (
    _complex_binary("array.add", np.add, Real(), Real()),
    _complex_binary("array.subtract", np.subtract, Real(), Real()),
    _complex_binary("array.multiply", np.multiply, Real(), Real()),
    _complex_binary(
        "array.divide",
        np.divide,
        Real(),
        Nonzero(),
        variants=_COMPLEX_DIVIDE_VARIANTS,
    ),
)

# --- reductions and scans ----------------------------------------------------

_REDUCTIONS: tuple[InvocationCase, ...] = (
    _reduction("array.sum", np.sum, Real()),
    _reduction("array.mean", np.mean, Real()),
    _reduction(
        "array.prod",
        np.prod,
        Nonzero(),
        variants=_PRODUCT_REDUCTION_VARIANTS,
        numerical_reference=NumericalReference.COMPLEX_STEP,
    ),
    _reduction("array.cumsum", lambda x: np.cumsum(x, axis=-1), Real()),
    _reduction(
        "array.cumprod",
        lambda x: np.cumprod(x, axis=-1),
        Nonzero(),
        variants=_CUMPROD_REDUCTION_VARIANTS,
    ),
    _reduction("array.var", np.var, Distinct()),
    _reduction("array.std", np.std, Distinct()),
    _reduction("array_ext.nansum", np.nansum, Real()),
    _reduction("array_ext.nanmean", np.nanmean, Real()),
    _reduction("array_ext.nanstd", np.nanstd, Distinct()),
    _reduction("array_ext.nanvar", np.nanvar, Distinct()),
    _reduction("array_ext.nanprod", np.nanprod, Nonzero()),
    # Extrema select one element, so a tie is the kink; keep values separated.
    _reduction("array.max", np.max, Distinct()),
    _reduction("array.min", np.min, Distinct()),
    _reduction("array_ext.amax", np.amax, Distinct()),
    _reduction("array_ext.amin", np.amin, Distinct()),
    _reduction("array_ext.nanmax", np.nanmax, Distinct()),
    _reduction("array_ext.nanmin", np.nanmin, Distinct()),
    InvocationCase(
        op="array.cumsum",
        call=np.cumsum,
        arguments=(Argument("x", Real(), shape=_VECTOR),),
        variants=_FIXED_REAL_DTYPE_VARIANTS,
        reason="the default axis=None flattening contract is tested separately",
    ),
    InvocationCase(
        op="array.cumprod",
        call=np.cumprod,
        arguments=(Argument("x", Nonzero(), shape=_VECTOR),),
        variants=_CUMPROD_FIXED_REAL_DTYPE_VARIANTS,
        reason="the default axis=None flattening contract is tested separately",
    ),
    InvocationCase(
        op="array.sum",
        call=lambda x: np.sum(
            x,
            axis=1,
            dtype=np.float32,
            initial=np.float32(0.25),
            keepdims=True,
            where=np.array([[True, False, True], [True, True, False]]),
        ),
        arguments=(Argument("x", Real(), shape=(2, 3), dtype="float32"),),
        tolerance=_FLOAT32_TOLERANCE,
        reason=(
            "controlled float32 reduction covers where, initial, axis, and keepdims; "
            "functional masking may change the reduction order by one ulp"
        ),
    ),
    InvocationCase(
        op="array.mean",
        call=np.mean,
        arguments=(Argument("x", Real(), shape=(2, 3), dtype="float32"),),
        tolerance=_FLOAT32_TOLERANCE,
        reason="float32 input covers the native accumulator and cotangent dtype",
    ),
    InvocationCase(
        op="array.sum",
        call=lambda x: np.sum(x, axis=(0, 2)),
        arguments=(Argument("x", Real(), shape=(2, 3, 4), dtype="float32"),),
        tolerance=_FLOAT32_TOLERANCE,
        reason="a tuple axis covers multidimensional cotangent expansion in float32",
    ),
    InvocationCase(
        op="array.mean",
        call=lambda x: np.mean(x, axis=(0, 2), keepdims=True),
        arguments=(Argument("x", Real(), shape=(2, 3, 4), dtype="float32"),),
        tolerance=_FLOAT32_TOLERANCE,
        reason="tuple axes with keepdims cover mean's reduction shape and dtype contract",
    ),
    InvocationCase(
        op="array.var",
        call=np.var,
        arguments=(Argument("x", Real(), shape=(2, 3), dtype="float32"),),
        tolerance=_FLOAT32_TOLERANCE,
        reason="float32 input covers real reduction result and cotangent dtype",
    ),
)

_REDUCTIONS_COMPLEX: tuple[InvocationCase, ...] = (
    _complex_reduction("array.sum", np.sum, Real()),
    _complex_reduction("array.mean", np.mean, Real()),
    replace(
        _complex_reduction("array.prod", np.prod, Nonzero()),
        variants=_PRODUCT_COMPLEX_REDUCTION_VARIANTS,
    ),
    _complex_reduction("array.cumsum", lambda x: np.cumsum(x, axis=-1), Real()),
    _complex_reduction("array.cumprod", lambda x: np.cumprod(x, axis=-1), Nonzero()),
    _complex_reduction("array_ext.nansum", np.nansum, Real()),
    _complex_reduction("array_ext.nanmean", np.nanmean, Real()),
    replace(
        _complex_reduction("array_ext.nanprod", np.nanprod, Nonzero()),
        variants=_PRODUCT_COMPLEX_REDUCTION_VARIANTS,
    ),
)

# --- shape and layout --------------------------------------------------------

_SHAPE: tuple[InvocationCase, ...] = (
    *(
        replace(case, variants=_FIXED_NUMERIC_DTYPE_VARIANTS)
        for case in (
            _unary("array.reshape", lambda x: np.reshape(x, (2, 3)), Real()),
            _unary("array.transpose", lambda x: np.transpose(np.reshape(x, (2, 3))), Real()),
            _unary("array.swapaxes", lambda x: np.swapaxes(np.reshape(x, (2, 3)), 0, 1), Real()),
            _unary("array.moveaxis", lambda x: np.moveaxis(np.reshape(x, (2, 3)), 0, 1), Real()),
            _unary("array.expand_dims", lambda x: np.expand_dims(x, 0), Real()),
            _unary("array.squeeze", lambda x: np.squeeze(np.expand_dims(x, 0)), Real()),
            _unary("array.broadcast_to", lambda x: np.broadcast_to(x, (3, 6)), Real()),
            _unary("array_ext.ravel", np.ravel, Real()),
            _unary("array.flip", np.flip, Real()),
            _unary("array_ext.fliplr", lambda x: np.fliplr(np.reshape(x, (2, 3))), Real()),
            _unary("array_ext.flipud", lambda x: np.flipud(np.reshape(x, (2, 3))), Real()),
            _unary("array.roll", lambda x: np.roll(x, 2), Real()),
            _unary("array_ext.rot90", lambda x: np.rot90(np.reshape(x, (2, 3))), Real()),
            _unary(
                "array_ext.rollaxis",
                lambda x: np.rollaxis(np.reshape(x, (1, 2, 3)), 0, -1),
                Real(),
            ),
            _unary(
                "array.repeat",
                lambda x: _xp_call(x, "repeat", x, 2),
                Real(),
                frontend=Frontend.ARRAY_API,
            ),
            _unary(
                "array.tile",
                lambda x: _xp_call(x, "tile", x, (2,)),
                Real(),
                frontend=Frontend.ARRAY_API,
            ),
            _unary("array_ext.pad", lambda x: np.pad(x, (1, 2)), Real()),
            _unary("array.diff", lambda x: np.diff(x, n=0), Real()),
            _unary("array_ext.gradient", lambda x: np.gradient(x, edge_order=2), Real()),
            _unary("array.stack", lambda x: np.stack([x, x]), Real()),
            _unary("array.atleast_1d", np.atleast_1d, Real()),
            _unary("array.atleast_2d", np.atleast_2d, Real()),
            _unary("array.atleast_3d", np.atleast_3d, Real()),
            _unary("array_ext.diag", lambda x: np.diag(np.reshape(x[:4], (2, 2))), Real()),
            _unary(
                "array.diagonal",
                lambda x: np.linalg.diagonal(np.reshape(x[:4], (2, 2))),
                Real(),
            ),
            _unary("array.trace", lambda x: np.trace(np.reshape(x[:4], (2, 2))), Real()),
            _unary("array.tril", lambda x: np.tril(np.reshape(x[:4], (2, 2))), Real()),
            _unary(
                "array.triu",
                lambda x: _xp_call(
                    x,
                    "triu",
                    _xp_call(x, "reshape", x[:4], (2, 2)),
                ),
                Real(),
                frontend=Frontend.ARRAY_API,
            ),
            _unary("advect.getitem", lambda x: x[1:5], Real()),
            _unary("array_ext.nan_to_num", np.nan_to_num, Real()),
        )
    ),
    InvocationCase(
        op="array.trace",
        call=lambda x: np.trace(x, offset=1),
        arguments=(Argument("x", Real(), shape=(3, 4)),),
        reason="a nonzero offset covers trace's shifted-diagonal transpose",
    ),
    InvocationCase(
        op="array.diagonal",
        call=lambda x: np.linalg.diagonal(x, offset=-1),
        arguments=(Argument("x", Real(), shape=(3, 4)),),
        reason="a negative offset covers diagonal's shifted scatter transpose",
    ),
    _binary(
        "array.concatenate",
        lambda left, right: np.concatenate((left, right), axis=None),
        Real(),
        Real(),
        variants=(
            InputVariant(
                "distinct-shapes",
                shapes={"a": (2, 3), "b": (2, 2)},
            ),
        ),
        reason="axis=None flattens inputs and the pullback restores their distinct shapes",
    ),
    InvocationCase(
        op="array.diff",
        call=lambda x: np.diff(
            x,
            n=2,
            axis=1,
            prepend=np.full((3, 1), 1.0 + 2.0j),
        ),
        arguments=(Argument("x", Real(), shape=(3, 4), dtype="complex128"),),
        reason="second differences with a prepended constant cover the nondefault stencil",
    ),
    InvocationCase(
        op="array_ext.pad",
        call=lambda x: np.pad(x, ((1, 2), (2, 0))),
        arguments=(Argument("x", Real(), shape=(3, 4)),),
        reason="asymmetric per-axis widths cover multidimensional padding transposition",
    ),
    InvocationCase(
        op="array.repeat",
        call=lambda x: _xp_call(x, "repeat", x, 3, axis=1),
        arguments=(Argument("x", Real(), shape=(3, 4)),),
        frontend=Frontend.ARRAY_API,
        reason="axis-preserving repetition has a distinct transpose from flattened repeat",
    ),
    InvocationCase(
        op="array.tile",
        call=lambda x: _xp_call(x, "tile", x, (2, 3)),
        arguments=(Argument("x", Real(), shape=(3, 4)),),
        frontend=Frontend.ARRAY_API,
        reason="per-axis repetitions cover multidimensional tile reduction",
    ),
    *(
        InvocationCase(
            op="array_ext.gradient",
            call=lambda x, axis=axis: np.gradient(x, axis=axis),
            arguments=(Argument("x", Real(), shape=(3, 4)),),
            reason=(
                f"axis={axis} covers the corresponding multidimensional finite-difference transpose"
            ),
        )
        for axis in (0, 1)
    ),
)

# --- contractions ------------------------------------------------------------


def _new_numpy_contractions() -> tuple[InvocationCase, ...]:
    matvec = getattr(np, "matvec", None)
    vecmat = getattr(np, "vecmat", None)
    if not isinstance(matvec, np.ufunc) or not isinstance(vecmat, np.ufunc):
        return ()
    return (
        InvocationCase(
            op="array_ext.matvec",
            call=matvec,
            arguments=(
                Argument("a", Real(), shape=(3, 4)),
                Argument("b", Real(), shape=(4,)),
            ),
        ),
        InvocationCase(
            op="array_ext.vecmat",
            call=vecmat,
            arguments=(
                Argument("a", Real(), shape=(3,)),
                Argument("b", Real(), shape=(3, 4)),
            ),
        ),
        InvocationCase(
            op="array_ext.matvec",
            call=matvec,
            arguments=(
                Argument("a", Real(), shape=(3, 4), dtype="complex128"),
                Argument("b", Real(), shape=(4,), dtype="complex128"),
            ),
            reason="complex contraction exercises conjugation-free matrix-vector transposition",
        ),
        InvocationCase(
            op="array_ext.vecmat",
            call=vecmat,
            arguments=(
                Argument("a", Real(), shape=(3,), dtype="complex128"),
                Argument("b", Real(), shape=(3, 4), dtype="complex128"),
            ),
            reason="complex contraction exercises real-adjoint vector-matrix transposition",
        ),
    )


def _second_order_signal_cases(op: str, operation: Any) -> tuple[InvocationCase, ...]:
    return tuple(
        InvocationCase(
            op=op,
            call=lambda x, mode=mode: np.sum(operation(x, x, mode=mode) ** 2),
            arguments=(Argument("x", Real(), shape=(4,)),),
            laws=DEFAULT_LAWS | {Law.SECOND_ORDER},
            reason=f"mode={mode} reuses one operand and requires a traceable nonzero Hessian",
        )
        for mode in ("full", "same", "valid")
    )


_CONTRACTIONS: tuple[InvocationCase, ...] = (
    InvocationCase(
        op="array.matmul",
        call=np.matmul,
        arguments=(
            Argument("a", Real(), shape=(3,)),
            Argument("b", Real(), shape=(3,)),
        ),
        variants=_MATMUL_VARIANTS,
    ),
    *_new_numpy_contractions(),
    _binary(
        "array_ext.dot",
        np.dot,
        Real(),
        Real(),
        variants=(
            *_CONTRACTION_VARIANTS,
            InputVariant(
                "scalar-vector-complex128",
                shapes={"a": (), "b": (2,)},
                dtypes={"a": "complex128", "b": "complex128"},
            ),
            InputVariant(
                "vector-scalar-complex128",
                shapes={"a": (2,), "b": ()},
                dtypes={"a": "complex128", "b": "complex128"},
            ),
        ),
    ),
    _binary(
        "array_ext.inner",
        np.inner,
        Real(),
        Real(),
        variants=_CONTRACTION_VARIANTS,
    ),
    InvocationCase(
        op="array_ext.inner",
        call=np.inner,
        arguments=(
            Argument("a", Real(), shape=(2, 3, 4)),
            Argument("b", Real(), shape=(5, 4)),
        ),
        reason="higher-rank inner products cover independent leading dimensions",
    ),
    _binary("array.outer", np.outer, Real(), Real()),
    InvocationCase(
        op="array.outer",
        call=np.outer,
        arguments=(
            Argument("a", Real(), shape=(2, 3)),
            Argument("b", Real(), shape=(4,)),
        ),
        reason="outer flattens a non-vector operand before forming the product",
    ),
    _binary(
        "array_ext.kron",
        np.kron,
        Real(),
        Real(),
        variants=_CONTRACTION_VARIANTS,
    ),
    InvocationCase(
        op="array_ext.kron",
        call=np.kron,
        arguments=(
            Argument("a", Real(), shape=(2, 3)),
            Argument("b", Real(), shape=(2, 2)),
        ),
        reason="matrix Kronecker products cover per-axis block expansion",
    ),
    _binary("array.tensordot", lambda a, b: np.tensordot(a, b, axes=1), Real(), Real()),
    _binary("array_ext.einsum", lambda a, b: np.einsum("i,i->", a, b), Real(), Real()),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a, b: np.einsum("ij,jk", a, b),
        arguments=(
            Argument("a", Real(), shape=(2, 3)),
            Argument("b", Real(), shape=(3, 4)),
        ),
        reason="implicit output notation is normalized before differentiation",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a: np.einsum("ii->", a),
        arguments=(Argument("a", Real(), shape=(4, 4)),),
        reason="a repeated operand label lowers through a diagonal",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a: np.einsum("ii->i", a),
        arguments=(Argument("a", Real(), shape=(4, 4)),),
        reason="a repeated operand label may survive in the output",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a: np.sum(np.einsum("ii->i", a) ** 3),
        arguments=(Argument("a", Real(), shape=(3, 3)),),
        laws=DEFAULT_LAWS | {Law.SECOND_ORDER},
        reason="a diagonal contraction composed with a cubic requires traceable second order",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a: np.einsum("ij->i", a),
        arguments=(Argument("a", Real(), shape=(3, 4)),),
        reason="an operand-local contraction label lowers through a reduction",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a: np.einsum("ij->", a),
        arguments=(Argument("a", Real(), shape=(3, 4)),),
        reason="a fully reduced operand lowers to a scalar contraction input",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a, b: np.einsum("...ij,...jk->...ik", a, b),
        arguments=(
            Argument("a", Real(), shape=(2, 1, 2, 3), dtype="complex128"),
            Argument("b", Real(), shape=(4, 3, 2), dtype="complex128"),
        ),
        reason="ellipsis expansion covers broadcast batch dimensions and complex adjoints",
    ),
    InvocationCase(
        op="array_ext.einsum",
        call=lambda a, b: np.einsum("iij,jk->ik", a, b),
        arguments=(
            Argument("a", Real(), shape=(2, 2, 3)),
            Argument("b", Real(), shape=(3, 4)),
        ),
        reason="diagonal extraction composes with an ordinary contraction",
    ),
    *(
        _binary(
            "array_ext.convolve",
            lambda a, b, mode=mode: np.convolve(a, b, mode=mode),
            Real(),
            Real(),
            variants=_SIGNAL_VARIANTS,
        )
        for mode in ("full", "same", "valid")
    ),
    *(
        _binary(
            "array_ext.correlate",
            lambda a, b, mode=mode: np.correlate(a, b, mode=mode),
            Real(),
            Real(),
            variants=_SIGNAL_VARIANTS,
        )
        for mode in ("full", "same", "valid")
    ),
    *_second_order_signal_cases("array_ext.convolve", np.convolve),
    *_second_order_signal_cases("array_ext.correlate", np.correlate),
    InvocationCase(
        op="array.cross",
        call=np.cross,
        arguments=(
            Argument("a", Real(), shape=(3,)),
            Argument("b", Real(), shape=(3,)),
        ),
    ),
    InvocationCase(
        op="array.cross",
        call=np.cross,
        arguments=(
            Argument("a", Real(), shape=(2, 3)),
            Argument("b", Real(), shape=(2, 3)),
        ),
        reason="batched three-vectors cover cross-product leading dimensions",
    ),
)

# --- linear algebra ----------------------------------------------------------

_LINALG: tuple[InvocationCase, ...] = (
    InvocationCase(
        op="array_ext.linalg.solve",
        call=np.linalg.solve,
        arguments=(
            Argument("a", WellConditioned(), shape=_MATRIX),
            Argument("b", Real(), shape=(4,)),
        ),
        tolerance=_LINALG_TOLERANCE,
        variants=_SOLVE_VARIANTS,
    ),
    _matrix("array_ext.linalg.inv", np.linalg.inv, WellConditioned(), tolerance=_LINALG_TOLERANCE),
    _matrix("array_ext.linalg.det", np.linalg.det, WellConditioned(), tolerance=_LINALG_TOLERANCE),
    _matrix(
        "array_ext.linalg.slogdet",
        lambda a: _xp_call(a, "linalg.slogdet", a)[1],
        WellConditioned(),
        frontend=Frontend.ARRAY_API,
        tolerance=_LINALG_TOLERANCE,
    ),
    _matrix(
        "array_ext.linalg.pinv",
        np.linalg.pinv,
        WellConditioned(),
        tolerance=_LINALG_TOLERANCE,
    ),
    InvocationCase(
        op="array_ext.linalg.pinv",
        call=np.linalg.pinv,
        arguments=(Argument("a", WellConditioned(), shape=(4, 2)),),
        tolerance=_LINALG_TOLERANCE,
        variants=_RECTANGULAR_MATRIX_VARIANTS,
        reason="tall and wide matrices exercise both pseudoinverse shape branches",
    ),
    _matrix(
        "array_ext.linalg.cholesky",
        np.linalg.cholesky,
        SymmetricPositiveDefinite(),
        tolerance=_LINALG_TOLERANCE,
    ),
    _matrix(
        "array_ext.linalg.eigvalsh",
        np.linalg.eigvalsh,
        SymmetricPositiveDefinite(),
        tolerance=_LINALG_TOLERANCE,
    ),
    _matrix(
        "array_ext.linalg.eigh",
        lambda a: np.linalg.eigh(a)[0],
        SymmetricPositiveDefinite(),
        tolerance=_LINALG_TOLERANCE,
    ),
    InvocationCase(
        op="array_ext.linalg.eigh",
        call=np.linalg.eigh,
        arguments=(Argument("a", SymmetricPositiveDefinite(), shape=_MATRIX),),
        variants=_SQUARE_MATRIX_VARIANTS,
        tolerance=_LINALG_TOLERANCE,
        reason="full eigensystems exercise the phase-aligned eigenvector derivative and adjoint",
    ),
    _matrix(
        "array_ext.linalg.svdvals",
        np.linalg.svdvals,
        WellConditioned(),
        tolerance=_LINALG_TOLERANCE,
    ),
    InvocationCase(
        op="array_ext.linalg.svdvals",
        call=np.linalg.svdvals,
        arguments=(Argument("a", WellConditioned(), shape=(4, 2)),),
        tolerance=_LINALG_TOLERANCE,
        variants=_RECTANGULAR_MATRIX_VARIANTS,
        reason="tall and wide matrices exercise both reduced singular-value branches",
    ),
    _matrix(
        "array_ext.linalg.svd",
        lambda a: np.linalg.svd(a)[1],
        WellConditioned(),
        tolerance=_LINALG_TOLERANCE,
    ),
    InvocationCase(
        op="array_ext.linalg.svd",
        call=lambda a: np.linalg.svd(a, full_matrices=False)[1],
        arguments=(Argument("a", WellConditioned(), shape=(4, 2)),),
        tolerance=_LINALG_TOLERANCE,
        variants=_RECTANGULAR_MATRIX_VARIANTS,
        reason="reduced tall and wide SVD covers non-square derivative branches",
    ),
    InvocationCase(
        op="array_ext.linalg.svd",
        call=lambda a: np.linalg.svd(a, full_matrices=False),
        arguments=(Argument("a", WellConditioned(), shape=(4, 2)),),
        tolerance=_LINALG_TOLERANCE,
        variants=_RECTANGULAR_MATRIX_VARIANTS,
        reason="full reduced SVD output exercises both singular-vector phase gauges and adjoints",
    ),
    _matrix(
        "array_ext.linalg.qr",
        lambda a: _xp_call(a, "linalg.qr", a)[1],
        WellConditioned(),
        frontend=Frontend.ARRAY_API,
        tolerance=_LINALG_TOLERANCE,
    ),
    _matrix(
        "array_ext.linalg.qr_r",
        lambda a: np.linalg.qr(a, mode="r"),
        WellConditioned(),
        tolerance=_LINALG_TOLERANCE,
    ),
    InvocationCase(
        op="array_ext.linalg.qr",
        call=lambda a: tuple(_xp_call(a, "linalg.qr", a)),
        arguments=(Argument("a", WellConditioned(), shape=(3, 5)),),
        frontend=Frontend.ARRAY_API,
        tolerance=_LINALG_TOLERANCE,
        reason="wide reduced QR covers the full-row-rank gauge and both outputs",
    ),
    InvocationCase(
        op="array_ext.linalg.qr_r",
        call=lambda a: np.linalg.qr(a, mode="r"),
        arguments=(Argument("a", WellConditioned(), shape=(3, 5)),),
        tolerance=_LINALG_TOLERANCE,
        reason="wide R-only QR shares the full-row-rank derivative path",
    ),
    InvocationCase(
        op="array_ext.linalg.eig",
        call=lambda a: np.linalg.eig(a)[0],
        arguments=(Argument("a", StableEigensystem(), shape=_MATRIX),),
        variants=_REAL_EIGEN_MATRIX_VARIANTS,
        tolerance=_LINALG_TOLERANCE,
        reason=("real-input eig remains dynamic because NumPy's output dtype is data-dependent"),
    ),
    InvocationCase(
        op="array_ext.linalg.eig",
        call=lambda a: np.linalg.eig(a)[0],
        arguments=(Argument("a", StableEigensystem(), shape=_MATRIX, dtype="complex128"),),
        laws=DEFAULT_LAWS | {Law.STAGED},
        variants=_COMPLEX_EIGEN_MATRIX_VARIANTS,
        tolerance=_LINALG_TOLERANCE,
        reason="complex-input eig has a data-independent staged output dtype",
    ),
    InvocationCase(
        op="array_ext.linalg.eig",
        call=np.linalg.eig,
        arguments=(Argument("a", StableEigensystem(), shape=(3, 3), dtype="complex128"),),
        tolerance=_LINALG_TOLERANCE,
        reason="full complex eigensystems exercise NumPy's eigenvector phase gauge and adjoint",
    ),
    InvocationCase(
        op="array_ext.linalg.eigvals",
        call=np.linalg.eigvals,
        arguments=(Argument("a", StableEigensystem(), shape=_MATRIX),),
        variants=_REAL_EIGEN_MATRIX_VARIANTS,
        tolerance=_LINALG_TOLERANCE,
        reason=(
            "real-input eigvals remains dynamic because NumPy's output dtype is data-dependent"
        ),
    ),
    InvocationCase(
        op="array_ext.linalg.eigvals",
        call=np.linalg.eigvals,
        arguments=(Argument("a", StableEigensystem(), shape=_MATRIX, dtype="complex128"),),
        laws=DEFAULT_LAWS | {Law.STAGED},
        variants=_COMPLEX_EIGEN_MATRIX_VARIANTS,
        tolerance=_LINALG_TOLERANCE,
        reason="complex-input eigvals has a data-independent staged output dtype",
    ),
    _unary("array_ext.linalg.norm", np.linalg.norm, Nonzero()),
    InvocationCase(
        op="array_ext.linalg.norm",
        call=np.linalg.norm,
        arguments=(Argument("x", Nonzero(), shape=(2, 3, 2)),),
        reason="rank-three default norm flattens every axis before differentiation",
    ),
)

# --- spectral ----------------------------------------------------------------


def _spectral(op: str, call: Any) -> InvocationCase:
    return InvocationCase(op=op, call=call, arguments=(Argument("x", Real(), shape=(8,)),))


def _plane(x: Any) -> Any:
    return np.reshape(x, (2, 4))


def _fft_signature_cases() -> tuple[InvocationCase, ...]:
    cases = [
        InvocationCase(
            op=op,
            call=transform,
            arguments=(Argument("x", Real(), shape=(4,), dtype="complex128"),),
            static={"n": length, "norm": norm},
            reason=(
                f"complex n={length}, norm={norm!r} covers the real adjoint "
                "for both truncation and zero-padding"
            ),
        )
        for op, transform in (
            ("array_ext.fft.fft", np.fft.fft),
            ("array_ext.fft.ifft", np.fft.ifft),
        )
        for norm in (None, "forward", "ortho")
        for length in (3, 6)
    ]

    cases.extend(
        (
            InvocationCase(
                op="array_ext.fft.rfft",
                call=np.fft.rfft,
                arguments=(Argument("x", Real(), shape=(5,)),),
                reason="the complex half-spectrum exercises arbitrary complex cotangents",
            ),
            InvocationCase(
                op="array_ext.fft.irfft",
                call=np.fft.irfft,
                arguments=(Argument("x", Real(), shape=(3,), dtype="complex128"),),
                static={"n": 5},
                reason="a complex half-spectrum exercises the inverse real-FFT adjoint",
            ),
        )
    )

    for op, transform, shape, size, axes in (
        ("array_ext.fft.rfft2", np.fft.rfft2, (3, 4), (4, 5), None),
        ("array_ext.fft.rfftn", np.fft.rfftn, (3, 4), (4, 5), (0, 1)),
        ("array_ext.fft.irfft2", np.fft.irfft2, (4, 3), (3, 6), None),
        ("array_ext.fft.irfftn", np.fft.irfftn, (4, 3), (3, 6), (0, 1)),
    ):
        dtype = "float64" if op in {"array_ext.fft.rfft2", "array_ext.fft.rfftn"} else "complex128"
        for norm in (None, "forward", "ortho"):
            static: dict[str, Any] = {"s": size, "norm": norm}
            if axes is not None:
                static["axes"] = axes
            cases.append(
                InvocationCase(
                    op=op,
                    call=transform,
                    arguments=(Argument("x", Real(), shape=shape, dtype=dtype),),
                    static=static,
                    reason=(
                        f"multidimensional real transform with norm={norm!r} covers "
                        "resize, axes, and real-inner-product transposition"
                    ),
                )
            )
    return tuple(cases)


_FFT: tuple[InvocationCase, ...] = (
    _spectral("array_ext.fft.fft", lambda x: np.real(np.fft.fft(x))),
    _spectral("array_ext.fft.ifft", lambda x: np.real(np.fft.ifft(x))),
    _spectral("array_ext.fft.rfft", lambda x: np.real(np.fft.rfft(x))),
    _spectral("array_ext.fft.irfft", lambda x: np.fft.irfft(np.fft.rfft(x))),
    _spectral("array_ext.fft.fft2", lambda x: np.real(np.fft.fft2(_plane(x)))),
    _spectral("array_ext.fft.ifft2", lambda x: np.real(np.fft.ifft2(_plane(x)))),
    _spectral("array_ext.fft.fftn", lambda x: np.real(np.fft.fftn(_plane(x)))),
    _spectral("array_ext.fft.ifftn", lambda x: np.real(np.fft.ifftn(_plane(x)))),
    _spectral("array_ext.fft.rfft2", lambda x: np.real(np.fft.rfft2(_plane(x)))),
    _spectral("array_ext.fft.rfftn", lambda x: np.real(np.fft.rfftn(_plane(x)))),
    _spectral("array_ext.fft.irfft2", lambda x: np.fft.irfft2(np.fft.rfft2(_plane(x)))),
    _spectral("array_ext.fft.irfftn", lambda x: np.fft.irfftn(np.fft.rfftn(_plane(x)))),
    InvocationCase(
        op="array_ext.fft.hfft",
        call=lambda x: np.fft.hfft(x, n=8),
        arguments=(Argument("x", Real(), shape=(5,), dtype="complex128"),),
    ),
    _spectral("array_ext.fft.ihfft", lambda x: np.fft.ihfft(x, n=8)),
    _spectral("array_ext.fft.fftshift", np.fft.fftshift),
    _spectral("array_ext.fft.ifftshift", np.fft.ifftshift),
    *_fft_signature_cases(),
)

# --- interpolation, selection, and sampling ----------------------------------

_SELECTION: tuple[InvocationCase, ...] = (
    _unary(
        "array_ext.bincount",
        lambda weights: np.bincount(
            np.array([0, 2, 1, 2, 0, 2]),
            weights=weights,
            minlength=4,
        ),
        Real(),
    ),
    InvocationCase(
        op="array_ext.interp",
        call=np.interp,
        arguments=(
            # ``Interior`` keeps queries off the knots, where a piecewise linear
            # map is kinked, while still exercising the clamped branch.
            Argument("x", Interior("xp"), shape=(5,)),
            Argument("xp", Increasing(), shape=(6,)),
            Argument("fp", Distinct(), shape=(6,)),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0, 1, 2}),
        reason="the domain guarantees one in-range query and nonzero interpolation slopes",
    ),
    InvocationCase(
        op="array_ext.interp",
        call=lambda grid, values: np.interp(np.array([-0.25, 0.4, 2.0]), grid, values),
        arguments=(
            Argument("grid", SpanningGrid(), shape=(5,)),
            Argument("values", Distinct(), shape=(5,)),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0, 1}),
        reason="the fixed query contains an interior point and values have nonzero slopes",
    ),
    InvocationCase(
        op="array_ext.interp",
        call=lambda values: np.interp(
            np.array([-1.0, 0.5, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            values,
        ),
        arguments=(Argument("values", Distinct(), shape=(3,)),),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0}),
        reason="default clamping routes below- and above-grid cotangents to endpoint values",
    ),
    InvocationCase(
        op="array_ext.interp",
        call=lambda values: np.interp(
            np.array([-1.0, 0.5, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            values,
            left=-5.0,
            right=7.0,
        ),
        arguments=(Argument("values", Distinct(), shape=(3,)),),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0}),
        reason="explicit boundary constants prevent out-of-range cotangents reaching values",
    ),
    InvocationCase(
        op="array_ext.interp",
        call=lambda values: np.interp(
            np.array([-1.0, 0.5, 2.0]),
            np.array([0.5]),
            values,
        ),
        arguments=(Argument("values", Real(), shape=(1,)),),
        reason="a single sample has no slope and every query selects that value",
    ),
    InvocationCase(
        op="array.linspace",
        call=lambda start, stop: np.linspace(start, stop, 5),
        arguments=(
            Argument("start", Real(), shape=()),
            Argument("stop", Real(), shape=()),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0, 1}),
        reason="each endpoint contributes to every nondegenerate linspace",
    ),
    InvocationCase(
        op="array.linspace",
        # The positional form is the one that regressed: ``num`` arrived as
        # args[2] and was silently replaced by NumPy's default of 50.
        call=lambda start, stop: np.sum(np.linspace(start, stop, 7, False)),  # noqa: FBT003
        arguments=(
            Argument("start", Real(), shape=()),
            Argument("stop", Real(), shape=()),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0, 1}),
        reason="the reduced positional form depends on both endpoints",
    ),
    InvocationCase(
        op="array.linspace",
        call=lambda start, stop: np.linspace(
            start,
            stop,
            num=4,
            endpoint=False,
            axis=1,
        ),
        arguments=(
            Argument("start", Real(), shape=(2,)),
            Argument("stop", Real(), shape=(2,)),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0, 1}),
        reason="array endpoints with axis=1 cover broadcast reconstruction and endpoint=False",
    ),
    InvocationCase(
        op="array.linspace",
        call=lambda start, stop: np.linspace(start, stop, num=1, endpoint=True),
        arguments=(
            Argument("start", Real(), shape=()),
            Argument("stop", Real(), shape=()),
        ),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0}),
        reason="a single endpoint depends only on start",
    ),
    InvocationCase(
        op="array.clip",
        # Bounds are chosen to sit inside the sample domain, so every draw has
        # both clamped and unclamped entries. A domain that can put every value
        # outside the interval would make the gradient legitimately zero and
        # the DEPENDENCE law would -- correctly -- reject the case.
        call=lambda x: np.clip(x, -0.31, 0.31),
        arguments=(Argument("x", ClipRegions(-0.31, 0.31), shape=_VECTOR),),
        laws=DEFAULT_LAWS | {Law.DEPENDENCE},
        dependence_indices=frozenset({0}),
        reason="the separated domain always leaves at least one value inside the clip interval",
    ),
    InvocationCase(
        op="array.where",
        call=lambda a, b: np.where(np.arange(6) % 2 == 0, a, b),
        arguments=(
            Argument("a", Real(), shape=_VECTOR),
            Argument("b", Real(), shape=_VECTOR),
        ),
    ),
)


def _index_update(x: Any) -> Any:
    """Exercise functionalised mutation through the same battery."""
    out = x.copy()
    out[1:4] += 0.5 * x[0:3]
    return out


_INTERNAL: tuple[InvocationCase, ...] = (
    InvocationCase(
        op="advect.index_update",
        call=_index_update,
        arguments=(Argument("x", Real(), shape=_VECTOR),),
    ),
    InvocationCase(
        op="advect.copy",
        call=lambda x: x.copy() * 2.0,
        arguments=(Argument("x", Real(), shape=_VECTOR),),
    ),
)

# --- multi-output and integer-boundary primitives ----------------------------

_MULTI_OUTPUT: tuple[InvocationCase, ...] = (
    InvocationCase(
        op="array_ext.modf",
        call=lambda x: np.modf(x)[0],
        arguments=(Argument("x", _BETWEEN_INTEGERS, shape=_VECTOR),),
        reason="only the fractional part carries derivative; the integral part is flat",
    ),
    InvocationCase(
        op="array_ext.divmod",
        call=lambda a, b: np.divmod(a, b)[1],
        arguments=(
            Argument("a", _MODULO_DIVIDEND, shape=_VECTOR),
            Argument("b", _MODULO_DIVISOR, shape=_VECTOR),
        ),
        reason="the quotient is flat; the remainder carries the derivative",
    ),
    InvocationCase(
        op="array_ext.frexp",
        call=lambda x: np.frexp(x)[0],
        arguments=(Argument("x", Positive(low=0.6, high=0.9), shape=_VECTOR),),
        reason="domain stays inside one binary exponent so the mantissa is smooth",
    ),
)


# --- Array API frontend invocations ------------------------------------------
# These exercise a genuinely different binder from NumPy. The portable matrix
# covers every currently qualified Array API operation with a differentiable
# array operand. Constant-only ``linspace`` is portable execution, but not an
# Array API derivative entrypoint because that frontend's endpoints are static
# attributes. NumPy's separate invocation contracts above differentiate them.


def _namespace(op: str, call: Any, *arguments: Argument, **kwargs: Any) -> InvocationCase:
    kwargs.setdefault("laws", DEFAULT_LAWS | {Law.STAGED})
    kwargs.setdefault("reason", "Array API invocation supports durable staged round trips")
    return InvocationCase(
        op=op,
        call=call,
        arguments=arguments,
        frontend=Frontend.ARRAY_API,
        **kwargs,
    )


def _namespace_unary(
    op: str,
    path: str,
    domain: Any,
    *,
    shape: tuple[int, ...] = _VECTOR,
    dtype: str = "float64",
    **kwargs: Any,
) -> InvocationCase:
    if shape == _VECTOR and dtype == "float64":
        kwargs.setdefault("variants", _POINTWISE_REAL_VARIANTS)
    return _namespace(
        op,
        lambda x: _xp_call(x, path, x),
        Argument("x", domain, shape=shape, dtype=dtype),
        **kwargs,
    )


def _namespace_binary(
    op: str,
    path: str,
    left: Any,
    right: Any,
    *,
    shape: tuple[int, ...] = _VECTOR,
    **kwargs: Any,
) -> InvocationCase:
    if shape == _VECTOR:
        kwargs.setdefault("variants", _POINTWISE_BINARY_VARIANTS)
    return _namespace(
        op,
        lambda a, b: _xp_call(a, path, a, b),
        Argument("a", left, shape=shape),
        Argument("b", right, shape=shape),
        **kwargs,
    )


PORTABLE_ARRAY_API_INVOCATIONS: tuple[InvocationCase, ...] = (
    _namespace_unary("array.absolute", "abs", Nonzero()),
    _namespace_binary("array.add", "add", Real(), Real()),
    _namespace_binary("array.arctan2", "atan2", Nonzero(), Nonzero()),
    _namespace(
        "array.clip",
        lambda x: _xp(x).clip(
            x,
            min=_xp(x).asarray(-0.31, dtype=x.dtype),
            max=_xp(x).asarray(0.31, dtype=x.dtype),
        ),
        Argument("x", ClipRegions(-0.31, 0.31), shape=_VECTOR),
        variants=_POINTWISE_REAL_VARIANTS,
    ),
    _namespace(
        "array.concatenate",
        lambda a, b: _xp(a).concat((a, b), axis=0),
        Argument("a", Real(), shape=_VECTOR),
        Argument("b", Real(), shape=_VECTOR),
        variants=_CONCATENATE_VARIANTS,
    ),
    _namespace_unary("array.cos", "cos", Real()),
    _namespace_binary("array.divide", "divide", Real(), Nonzero()),
    _namespace_unary("array.exp", "exp", Real()),
    _namespace_unary(
        "array_ext.fft.fft",
        "fft.fft",
        Real(),
        shape=(8,),
        dtype="complex128",
    ),
    _namespace_unary("array_ext.fft.rfft", "fft.rfft", Real(), shape=(8,)),
    _namespace_unary(
        "array_ext.linalg.det",
        "linalg.det",
        WellConditioned(),
        shape=_MATRIX,
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace_unary(
        "array_ext.linalg.eigvalsh",
        "linalg.eigvalsh",
        SymmetricPositiveDefinite(),
        shape=_MATRIX,
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace_unary(
        "array_ext.linalg.inv",
        "linalg.inv",
        WellConditioned(),
        shape=_MATRIX,
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace_unary(
        "array_ext.linalg.pinv",
        "linalg.pinv",
        WellConditioned(),
        shape=(4, 3),
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace(
        "array_ext.linalg.solve",
        lambda a, b: _xp(a).linalg.solve(a, b),
        Argument("a", WellConditioned(), shape=_MATRIX),
        Argument("b", Real(), shape=(4,)),
        tolerance=_LINALG_TOLERANCE,
        variants=_SOLVE_VARIANTS,
    ),
    _namespace_unary(
        "array_ext.linalg.svdvals",
        "linalg.svdvals",
        WellConditioned(),
        shape=(4, 3),
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace_unary("array.log", "log", Positive()),
    _namespace_binary(
        "array.matmul",
        "matmul",
        Real(),
        Real(),
        shape=_MATRIX,
    ),
    _namespace(
        "array.max",
        lambda x: _xp(x).max(x, axis=1),
        Argument("x", Distinct(), shape=(2, 3)),
    ),
    _namespace(
        "array.mean",
        lambda x: _xp(x).mean(x, axis=0, keepdims=True),
        Argument("x", Real(), shape=(2, 3)),
    ),
    _namespace_binary("array.multiply", "multiply", Real(), Real()),
    _namespace_unary("array.negative", "negative", Real()),
    _namespace(
        "array.transpose",
        lambda x: _xp(x).permute_dims(x, (1, 0)),
        Argument("x", Real(), shape=(2, 3)),
    ),
    _namespace(
        "array.reshape",
        lambda x: _xp(x).reshape(x, (2, 3)),
        Argument("x", Real(), shape=_VECTOR),
    ),
    _namespace_unary("array.sin", "sin", Real()),
    _namespace(
        "array.sort",
        lambda x: _xp(x).sort(x),
        Argument("x", Distinct(), shape=_VECTOR),
        variants=_REDUCTION_VARIANTS,
    ),
    _namespace_unary("array.sqrt", "sqrt", Positive()),
    _namespace_unary("array.square", "square", Real()),
    _namespace(
        "array.stack",
        lambda a, b: _xp(a).stack((a, b), axis=0),
        Argument("a", Real(), shape=_VECTOR),
        Argument("b", Real(), shape=_VECTOR),
        variants=_MATCHED_BINARY_VARIANTS,
    ),
    _namespace_binary("array.subtract", "subtract", Real(), Real()),
    _namespace(
        "array.sum",
        lambda x: _xp(x).sum(x, axis=1, keepdims=True),
        Argument("x", Real(), shape=(2, 3)),
    ),
    _namespace_unary("array.tanh", "tanh", Real()),
    _namespace(
        "array.where",
        lambda a, b: _xp(a).where(
            _xp(a).asarray([True, False, True, False, True, False]),
            a,
            b,
        ),
        Argument("a", Real(), shape=_VECTOR),
        Argument("b", Real(), shape=_VECTOR),
    ),
)


_OTHER_FRONTEND_INVOCATIONS: tuple[InvocationCase, ...] = (
    _namespace(
        "array.take",
        lambda x: _xp(x).take(x, _xp(x).asarray([0, 2, 2, 4])),
        Argument("x", Real(), shape=_VECTOR),
    ),
    InvocationCase(
        op="array.take",
        call=lambda x: np.take(x, np.array([0, 2, 2, 4])),
        arguments=(Argument("x", Real(), shape=_VECTOR),),
        reason="NumPy array-function binding is independent from the Array API binding",
    ),
    *(
        InvocationCase(
            op="array.take",
            call=lambda x, mode=mode: np.take(x, np.array([4, -1, 1]), axis=1, mode=mode),
            arguments=(Argument("x", Real(), shape=(2, 3, 4)),),
            reason=f"mode={mode} normalizes repeated indices on a middle axis",
        )
        for mode in ("wrap", "clip")
    ),
    _namespace(
        "array.take_along_axis",
        lambda x: _xp(x).take_along_axis(x, _xp(x).asarray([0, 2, 4])),
        Argument("x", Real(), shape=_VECTOR),
    ),
    InvocationCase(
        op="array.take_along_axis",
        call=lambda x: np.take_along_axis(x, np.array([0, 2, 4]), axis=0),
        arguments=(Argument("x", Real(), shape=_VECTOR),),
        reason="NumPy array-function binding is independent from the Array API binding",
    ),
    InvocationCase(
        op="array.take_along_axis",
        call=lambda x: np.take_along_axis(
            x,
            np.array([[0, 2], [1, 1]]),
            axis=1,
        ),
        arguments=(Argument("x", Real(), shape=(1, 3)),),
        reason="broadcast source dimensions reduce duplicate indexed cotangents",
    ),
    _namespace(
        "array.astype",
        lambda x: _xp(x).astype(x, _xp(x).float64),
        Argument("x", Real(), shape=_VECTOR),
    ),
    InvocationCase(
        op="array.astype",
        call=lambda x: x.astype(np.float32),
        arguments=(Argument("x", Real(), shape=_VECTOR),),
        tolerance=_CAST_TOLERANCE,
        reason="NumPy method binding is independent from the Array API binding",
    ),
    InvocationCase(
        op="array.sort",
        call=np.sort,
        arguments=(Argument("x", Distinct(), shape=_VECTOR),),
        reason="NumPy array-function binding is independent from the Array API binding",
    ),
    InvocationCase(
        "array_ext.partition",
        lambda x: np.partition(x, 2),
        (Argument("x", Distinct(), shape=_VECTOR),),
        reason="NumPy partition has no corresponding Array API operation",
    ),
    _namespace(
        "array.vecdot",
        lambda a, b: _xp(a).vecdot(a, b),
        Argument("a", Real(), shape=_VECTOR),
        Argument("b", Real(), shape=_VECTOR),
    ),
    _namespace(
        "array_ext.linalg.vector_norm",
        lambda x: _xp(x).linalg.vector_norm(x),
        Argument("x", Nonzero(), shape=_VECTOR),
        variants=_REDUCTION_VARIANTS,
    ),
    _namespace(
        "array_ext.linalg.vector_norm",
        lambda x: _xp(x).linalg.vector_norm(x, ord=3, axis=(0, 2), keepdims=True),
        Argument("x", Nonzero(), shape=(2, 2, 3)),
        reason="multi-axis p-norm covers tuple-axis normalization",
    ),
    _namespace(
        "array_ext.linalg.matrix_norm",
        lambda a: _xp(a).linalg.matrix_norm(a),
        Argument("a", WellConditioned(), shape=_MATRIX),
        tolerance=_LINALG_TOLERANCE,
    ),
    _namespace(
        "array_ext.linalg.matrix_norm",
        lambda a: _xp(a).linalg.matrix_norm(a, ord="nuc", keepdims=True),
        Argument("a", WellConditioned(), shape=(2, 2, 3)),
        tolerance=_LINALG_TOLERANCE,
        reason="nuclear norm covers the singular-value branch on batched matrices",
    ),
)


# --- SciPy special functions -------------------------------------------------
# ``advect.scipy`` registers these through the real ``advect.primitive`` path, so they
# exercise the custom-primitive path rather than the array-family tables. They
# are the closest thing in-tree to a third-party primitive, and they earn the
# same guarantees.


def _scipy_cases() -> tuple[InvocationCase, ...]:
    import advect.scipy  # noqa: F401, PLC0415 - registers the primitives
    from advect.scipy import ndimage, special  # noqa: PLC0415
    from advect.scipy._ndimage.selection import (  # noqa: PLC0415
        _selection_transpose_primitive,
    )
    from advect.scipy._ndimage.stencil import (  # noqa: PLC0415
        _stencil_input_transpose_primitive,
    )

    return (
        _unary(
            "custom.scipy.special.erf",
            special.erf,
            Real(),
        ),
        _unary(
            "custom.scipy.special.expit",
            special.expit,
            Real(),
        ),
        _unary(
            "custom.scipy.special.ndtr",
            special.ndtr,
            Real(),
        ),
        _unary("custom.scipy.special.log_ndtr", special.log_ndtr, Real()),
        _unary("custom.scipy.special.log_expit", special.log_expit, Real()),
        _unary("custom.scipy.special.erfc", special.erfc, Real()),
        _unary("custom.scipy.special.erfcx", special.erfcx, Real()),
        _unary("custom.scipy.special.erfinv", special.erfinv, Unit()),
        _unary(
            "custom.scipy.special.ndtri",
            special.ndtri,
            Positive(low=0.1, high=0.9),
        ),
        _unary(
            "custom.scipy.special.logsumexp",
            special.logsumexp,
            Real(),
        ),
        _unary(
            "custom.scipy.special.softmax",
            special.softmax,
            Real(),
        ),
        _unary(
            "custom.scipy.special.log_softmax",
            special.log_softmax,
            Real(),
        ),
        # digamma and gammaln have poles at the non-positive integers.
        _unary(
            "custom.scipy.special.digamma",
            special.digamma,
            Positive(),
        ),
        _unary(
            "custom.scipy.special.gammaln",
            special.gammaln,
            Positive(),
        ),
        InvocationCase(
            op="custom.scipy.special.polygamma",
            call=lambda x: special.polygamma(1, x),
            arguments=(Argument("x", Positive(), shape=_VECTOR),),
        ),
        InvocationCase(
            op="custom.scipy.special.polygamma",
            call=lambda n, x: special.polygamma(np.floor(n), x),
            arguments=(
                Argument(
                    "n",
                    Positive(),
                    shape=(2, 1),
                    differentiable=False,
                ),
                Argument("x", Positive(), shape=(1, 3)),
            ),
        ),
        _unary(
            "custom.scipy.ndimage.gaussian_filter",
            lambda x: ndimage.gaussian_filter(x, 0.8, mode="mirror", radius=2),
            Real(),
        ),
        _unary(
            "custom.scipy.ndimage.gaussian_filter1d",
            lambda x: ndimage.gaussian_filter1d(x, 0.9, mode="nearest", radius=2),
            Real(),
        ),
        _unary(
            "custom.scipy.ndimage.uniform_filter",
            lambda x: ndimage.uniform_filter(x, 3, mode="wrap"),
            Real(),
        ),
        _unary(
            "custom.scipy.ndimage.uniform_filter1d",
            lambda x: ndimage.uniform_filter1d(x, 3, mode="constant", cval=0.3),
            Real(),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.convolve",
            call=lambda x, weights: ndimage.convolve(x, weights, mode="wrap"),
            arguments=(
                Argument("x", Real(), shape=(3, 4)),
                Argument("weights", Real(), shape=(2, 2)),
            ),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.correlate",
            call=lambda x, weights: ndimage.correlate(x, weights, mode="nearest"),
            arguments=(
                Argument("x", Real(), shape=(3, 4)),
                Argument("weights", Real(), shape=(2, 2)),
            ),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.convolve1d",
            call=lambda x, weights: ndimage.convolve1d(x, weights, mode="mirror"),
            arguments=(
                Argument("x", Real(), shape=_VECTOR),
                Argument("weights", Real(), shape=(3,)),
            ),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.correlate1d",
            call=lambda x, weights: ndimage.correlate1d(x, weights, mode="constant", cval=0.2),
            arguments=(
                Argument("x", Real(), shape=_VECTOR),
                Argument("weights", Real(), shape=(3,)),
            ),
        ),
        _unary(
            "custom.scipy.ndimage.maximum_filter",
            lambda x: ndimage.maximum_filter(x, 3, mode="wrap"),
            Distinct(),
        ),
        _unary(
            "custom.scipy.ndimage.minimum_filter",
            lambda x: ndimage.minimum_filter(x, 3, mode="wrap"),
            Distinct(),
        ),
        _unary(
            "custom.scipy.ndimage.maximum_filter1d",
            lambda x: ndimage.maximum_filter1d(x, 3, mode="reflect"),
            Distinct(),
        ),
        _unary(
            "custom.scipy.ndimage.minimum_filter1d",
            lambda x: ndimage.minimum_filter1d(x, 3, mode="reflect"),
            Distinct(),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.grey_dilation",
            call=lambda x, structure: ndimage.grey_dilation(
                x,
                structure=structure,
                mode="wrap",
            ),
            arguments=(
                Argument("x", Distinct(gap=0.5), shape=_VECTOR),
                Argument("structure", Real(scale=0.01), shape=(3,)),
            ),
        ),
        InvocationCase(
            op="custom.scipy.ndimage.grey_erosion",
            call=lambda x, structure: ndimage.grey_erosion(
                x,
                structure=structure,
                mode="wrap",
            ),
            arguments=(
                Argument("x", Distinct(gap=0.5), shape=_VECTOR),
                Argument("structure", Real(scale=0.01), shape=(3,)),
            ),
        ),
        _unary(
            "custom.scipy.ndimage.median_filter",
            lambda x: ndimage.median_filter(x, size=3, mode="wrap"),
            Distinct(),
        ),
        _unary(
            "custom.scipy.ndimage.rank_filter",
            lambda x: ndimage.rank_filter(x, 1, size=3, mode="wrap"),
            Distinct(),
        ),
        _unary(
            "custom.scipy.ndimage.percentile_filter",
            lambda x: ndimage.percentile_filter(x, 50.0, size=3, mode="wrap"),
            Distinct(),
        ),
        InvocationCase(
            op="custom.scipy.ndimage._stencil_input_transpose",
            call=lambda x: _stencil_input_transpose_primitive(
                x,
                np.array([0.2, 0.6, 0.2]),
                axes=(0,),
                origins=(0,),
                modes=("reflect",),
                convolution=False,
            ),
            arguments=(Argument("x", Real(), shape=_VECTOR),),
        ),
        InvocationCase(
            op="custom.scipy.ndimage._selection_transpose",
            call=lambda x: _selection_transpose_primitive(
                x,
                np.array([0.2, -0.7, 1.3]),
                np.zeros(()),
                np.zeros(()),
                np.array([1.3, 1.3, 1.3]),
                axes=(0,),
                shape=(3,),
                footprint=(True, True, True),
                origins=(0,),
                modes=("wrap",),
                selection="maximum",
                dilation=False,
                rank=None,
                has_structure=False,
            ),
            arguments=(Argument("x", Real(), shape=(3,)),),
        ),
    )


_SCIPY: tuple[InvocationCase, ...] = _scipy_cases()


# Abstract rules for non-differentiable NumPy extensions still need executable
# staged coverage even though they do not belong in the derivative inventory.
STAGED_ONLY_INVOCATIONS: tuple[InvocationCase, ...] = (
    InvocationCase(
        op="array.signbit",
        call=np.signbit,
        arguments=(Argument("x", Real(), shape=_VECTOR, differentiable=False),),
        laws=frozenset({Law.STAGED}),
        reason="signbit is a staged, non-differentiable NumPy extension",
    ),
)

# Dynamic differentiation is intentionally wider than abstract staging. This
# closed set records exact invocation contracts whose current callable path has
# no complete abstract lowering; another frontend or static form of the same
# canonical operation may still earn a staged serialize/load law.
DYNAMIC_ONLY_STAGING_INVOCATIONS = frozenset(
    {
        "array.atleast_1d[numpy]",
        "array.atleast_2d[numpy]",
        "array.atleast_3d[numpy]",
        "array.linspace[numpy]",
        "array.linspace[numpy]#1",
        "array.linspace[numpy]#2",
        "array.linspace[numpy]#3",
        "array.outer[numpy]#1",
        "array.roll[numpy]",
        "array.swapaxes[numpy]",
        "array_ext.amax[numpy]",
        "array_ext.amin[numpy]",
        "array_ext.bincount[numpy]",
        "array_ext.cbrt[numpy]",
        "array_ext.deg2rad[numpy]",
        "array_ext.degrees[numpy]",
        "array_ext.diag[numpy]",
        "array_ext.divmod[numpy]",
        "array_ext.einsum[numpy]",
        "array_ext.einsum[numpy]#1",
        "array_ext.einsum[numpy]#2",
        "array_ext.einsum[numpy]#3",
        "array_ext.einsum[numpy]#4",
        "array_ext.einsum[numpy]#5",
        "array_ext.einsum[numpy]#6",
        "array_ext.einsum[numpy]#7",
        "array_ext.einsum[numpy]#8",
        "array_ext.fft.fft[numpy]#1",
        "array_ext.fft.fft[numpy]#3",
        "array_ext.fft.fft[numpy]#5",
        "array_ext.fft.ifft[numpy]#1",
        "array_ext.fft.ifft[numpy]#3",
        "array_ext.fft.ifft[numpy]#5",
        "array_ext.fft.irfft2[numpy]#1",
        "array_ext.fft.irfft2[numpy]#2",
        "array_ext.fft.irfft2[numpy]#3",
        "array_ext.fft.irfftn[numpy]#1",
        "array_ext.fft.irfftn[numpy]#2",
        "array_ext.fft.irfftn[numpy]#3",
        "array_ext.linalg.eig[numpy]",
        "array_ext.linalg.eig[numpy]#2",
        "array_ext.linalg.eigvals[numpy]",
        "array_ext.linalg.eigh[numpy]#1",
        "array_ext.linalg.norm[numpy]#1",
        "array_ext.linalg.svd[numpy]#2",
        "array_ext.exp2[numpy]",
        "array_ext.fabs[numpy]",
        "array_ext.flipud[numpy]",
        "array_ext.fliplr[numpy]",
        "array_ext.float_power[numpy]",
        "array_ext.fmax[numpy]",
        "array_ext.fmin[numpy]",
        "array_ext.fmod[numpy]",
        "array_ext.frexp[numpy]",
        "array_ext.inner[numpy]",
        "array_ext.inner[numpy]#1",
        "array_ext.interp[numpy]",
        "array_ext.interp[numpy]#1",
        "array_ext.interp[numpy]#2",
        "array_ext.interp[numpy]#3",
        "array_ext.interp[numpy]#4",
        "array_ext.kron[numpy]",
        "array_ext.kron[numpy]#1",
        "array_ext.logaddexp2[numpy]",
        "array_ext.modf[numpy]",
        "array_ext.nan_to_num[numpy]",
        "array_ext.pad[numpy]",
        "array_ext.pad[numpy]#1",
        "array_ext.partition[numpy]",
        "array_ext.rad2deg[numpy]",
        "array_ext.radians[numpy]",
        "array_ext.ravel[numpy]",
        "array_ext.rollaxis[numpy]",
        "array_ext.rot90[numpy]",
        "array_ext.sinc[numpy]",
    }
)


def _with_staged_contract(identifier: str, case: InvocationCase) -> InvocationCase:
    if identifier in DYNAMIC_ONLY_STAGING_INVOCATIONS or Law.STAGED in case.laws:
        return case
    return replace(
        case,
        laws=case.laws | {Law.STAGED},
        reason=case.reason or "this invocation has a complete abstract staged lowering",
    )


#: Every declared frontend invocation before its lifetime classification.
_RAW_BUILTIN_INVOCATIONS: tuple[InvocationCase, ...] = (
    *_ELEMENTWISE_REAL,
    *_ELEMENTWISE_COMPLEX,
    _ANGLE,
    _COMPLEX_SIGN,
    *_ELEMENTWISE_RESTRICTED,
    *_FLAT_ELEMENTWISE,
    *_BINARY,
    *_BINARY_COMPLEX,
    *_REDUCTIONS,
    *_REDUCTIONS_COMPLEX,
    *_SHAPE,
    *_CONTRACTIONS,
    *_LINALG,
    *_FFT,
    *_SELECTION,
    *_INTERNAL,
    *_MULTI_OUTPUT,
    *PORTABLE_ARRAY_API_INVOCATIONS,
    *_OTHER_FRONTEND_INVOCATIONS,
    *_SCIPY,
)


def _identified(
    invocations: tuple[InvocationCase, ...],
) -> tuple[tuple[str, InvocationCase], ...]:
    identified: list[tuple[str, InvocationCase]] = []
    seen: dict[str, int] = {}
    for case in invocations:
        count = seen.get(case.op, 0)
        seen[case.op] = count + 1
        suffix = "" if count == 0 else f"#{count}"
        identifier = f"{case.op}[{case.frontend.value}]{suffix}"
        identified.append((identifier, case))
    return tuple(identified)


_RAW_INVOCATIONS_BY_ID = dict(_identified(_RAW_BUILTIN_INVOCATIONS))
_unknown_dynamic_only = DYNAMIC_ONLY_STAGING_INVOCATIONS - _RAW_INVOCATIONS_BY_ID.keys()
if _unknown_dynamic_only:
    msg = f"dynamic-only staging classifications name unknown invocations: {_unknown_dynamic_only}"
    raise RuntimeError(msg)

INVOCATIONS_BY_ID: dict[str, InvocationCase] = {
    identifier: _with_staged_contract(identifier, case)
    for identifier, case in _RAW_INVOCATIONS_BY_ID.items()
}


#: Every declared frontend invocation with its exact lifetime contract.
BUILTIN_INVOCATIONS: tuple[InvocationCase, ...] = tuple(INVOCATIONS_BY_ID.values())
