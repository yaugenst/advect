# ruff: noqa: PLR2004
"""Abstract registrations and evaluators for linear algebra."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from advect.core._abstract_helpers import (
    broadcast_shape,
    dtype_kind_bits,
    dtype_name,
    promote_dtype,
    real_dtype,
    reduction_shape,
)
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array_ext.linalg.cholesky": rule("cholesky", 1, allowed=("upper",)),
    "array_ext.linalg.det": rule("det", 1),
    "array_ext.linalg.eig": rule("eig", 1),
    "array_ext.linalg.eigh": rule(
        "eigh",
        1,
        allowed=("UPLO",),
        generic_only=True,
    ),
    "array_ext.linalg.eigvals": rule("eigvals", 1),
    "array_ext.linalg.eigvalsh": rule("eigvalsh", 1, allowed=("UPLO",)),
    "array_ext.linalg.inv": rule("inv", 1),
    "array_ext.linalg.matrix_norm": rule(
        "matrix_norm",
        1,
        allowed=("keepdims", "ord"),
    ),
    "array_ext.linalg.norm": rule(
        "norm",
        1,
        positional=("ord", "axis", "keepdims"),
        allowed=("axis", "keepdims", "ord"),
    ),
    "array_ext.linalg.pinv": rule("pinv", 1, allowed=("hermitian",)),
    "array_ext.linalg.qr": rule(
        "qr",
        1,
        positional=("mode",),
        allowed=("mode",),
        generic_only=True,
    ),
    "array_ext.linalg.qr_r": rule(
        "qr_r",
        1,
        positional=("mode",),
        allowed=("mode",),
    ),
    "array_ext.linalg.slogdet": rule("slogdet", 1, generic_only=True),
    "array_ext.linalg.solve": rule("solve", 2),
    "array_ext.linalg.svd": rule(
        "svd",
        1,
        allowed=("compute_uv", "full_matrices", "hermitian"),
        generic_only=True,
    ),
    "array_ext.linalg.svdvals": rule("svdvals", 1),
    "array_ext.linalg.vector_norm": rule(
        "vector_norm",
        1,
        allowed=("axis", "keepdims", "ord"),
    ),
}


def _norm(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    axis = attrs.get("axis")
    if axis is None:
        if len(first.shape) not in {1, 2}:
            raise ValueError(
                "numpy.linalg.norm requires axis= for inputs with rank other than 1 or 2"
            )
    elif isinstance(axis, tuple) and len(axis) not in {1, 2}:
        raise ValueError("numpy.linalg.norm axis tuples must contain one or two axes")
    return (
        ArraySpec(
            reduction_shape(
                first.shape,
                axis,
                keepdims=bool(attrs.get("keepdims", False)),
            ),
            real_dtype(first.dtype),
        ),
    )


def _matrix_result(  # noqa: C901, PLR0911, PLR0912 - closed matrix shape cases
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if len(first.shape) < 2:
        raise ValueError(f"{kind} input must have at least two dimensions")
    batch_shape = first.shape[:-2]
    rows, columns = first.shape[-2:]
    if kind in {"cholesky", "det", "eigvals", "eigvalsh", "inv"} and rows != columns:
        raise ValueError(f"{kind} input must end in a square matrix")
    if kind == "cholesky":
        if type(attrs.get("upper", False)) is not bool:
            raise TypeError("cholesky upper must be a bool")
        return (ArraySpec(first.shape, dtype_name(first.dtype)),)
    if kind == "det":
        return (ArraySpec(batch_shape, dtype_name(first.dtype)),)
    if kind == "eigvals":
        dtype_kind, _bits = dtype_kind_bits(dtype_name(first.dtype))
        if dtype_kind != "complex":
            raise TypeError(
                "Staging numpy.linalg.eigvals requires a complex input because "
                "NumPy's output dtype for real matrices is data-dependent"
            )
        return (ArraySpec((*batch_shape, rows), dtype_name(first.dtype)),)
    if kind == "eigvalsh":
        uplo = attrs.get("UPLO", "L")
        if uplo not in {"L", "U"}:
            raise ValueError("eigvalsh UPLO must be 'L' or 'U'")
        return (ArraySpec((*batch_shape, rows), real_dtype(first.dtype)),)
    if kind == "inv":
        return (ArraySpec(first.shape, dtype_name(first.dtype)),)
    if kind == "matrix_norm":
        keepdims = attrs.get("keepdims", False)
        if type(keepdims) is not bool:
            raise TypeError("matrix_norm keepdims must be a bool")
        shape = (*batch_shape, 1, 1) if keepdims else batch_shape
        return (ArraySpec(shape, real_dtype(first.dtype)),)
    if kind == "pinv":
        if len(specs) == 2:
            tolerance_shape = specs[1].shape
            if broadcast_shape(tolerance_shape, batch_shape) != batch_shape:
                raise ValueError("pinv array tolerance must broadcast to the matrix batch shape")
        return (
            ArraySpec(
                (*batch_shape, columns, rows),
                dtype_name(first.dtype),
            ),
        )
    if kind == "qr_r":
        if attrs.get("mode", "r") != "r":
            raise ValueError("qr_r mode must be 'r'")
        return (
            ArraySpec(
                (*batch_shape, min(rows, columns), columns),
                dtype_name(first.dtype),
            ),
        )
    return (
        ArraySpec(
            (*batch_shape, min(rows, columns)),
            real_dtype(first.dtype),
        ),
    )


def _vector_norm(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    keepdims = attrs.get("keepdims", False)
    if type(keepdims) is not bool:
        raise TypeError("vector_norm keepdims must be a bool")
    return (
        ArraySpec(
            reduction_shape(
                first.shape,
                attrs.get("axis"),
                keepdims=keepdims,
            ),
            real_dtype(first.dtype),
        ),
    )


def _solve(
    specs: Sequence[ArraySpec],
    _attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    matrix_shape = specs[0].shape
    right_shape = specs[1].shape
    if len(matrix_shape) < 2 or matrix_shape[-2] != matrix_shape[-1]:
        raise ValueError("solve coefficient input must end in a square matrix")
    size = matrix_shape[-1]
    if len(right_shape) == 1:
        if right_shape[0] != size:
            raise ValueError("solve right-hand side has the wrong core dimension")
        shape = (*matrix_shape[:-2], size)
    else:
        if len(right_shape) < 2 or right_shape[-2] != size:
            raise ValueError("solve right-hand side has the wrong core dimension")
        shape = (
            *broadcast_shape(matrix_shape[:-2], right_shape[:-2]),
            size,
            right_shape[-1],
        )
    return (ArraySpec(shape, promote_dtype(specs)),)


def _decomposition(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[ArraySpec, ...]:
    first = specs[0]
    if len(first.shape) < 2:
        raise ValueError(f"{kind} input must have at least two dimensions")
    batch_shape = first.shape[:-2]
    rows, columns = first.shape[-2:]
    input_dtype = dtype_name(first.dtype)
    result_real_dtype = real_dtype(first.dtype)

    if kind in {"eig", "eigh", "slogdet"} and rows != columns:
        raise ValueError(f"{kind} input must end in a square matrix")
    if kind == "eig":
        dtype_kind, _bits = dtype_kind_bits(input_dtype)
        if dtype_kind != "complex":
            raise TypeError(
                "Staging numpy.linalg.eig requires a complex input because NumPy's "
                "output dtype for real matrices is data-dependent"
            )
        return (
            ArraySpec((*batch_shape, rows), input_dtype),
            ArraySpec(first.shape, input_dtype),
        )
    if kind == "eigh":
        return (
            ArraySpec((*batch_shape, rows), result_real_dtype),
            ArraySpec(first.shape, input_dtype),
        )
    if kind == "slogdet":
        return (
            ArraySpec(batch_shape, input_dtype),
            ArraySpec(batch_shape, result_real_dtype),
        )

    core_size = min(rows, columns)
    if kind == "qr":
        mode = attrs.get("mode", "reduced")
        if mode == "reduced":
            return (
                ArraySpec((*batch_shape, rows, core_size), input_dtype),
                ArraySpec((*batch_shape, core_size, columns), input_dtype),
            )
        if mode == "complete":
            return (
                ArraySpec((*batch_shape, rows, rows), input_dtype),
                ArraySpec((*batch_shape, rows, columns), input_dtype),
            )
        raise ValueError("qr mode must be 'reduced' or 'complete'")

    compute_uv = attrs.get("compute_uv", True)
    if compute_uv is not True:
        raise ValueError("svd compute_uv must be true for the three-output operation")
    hermitian = attrs.get("hermitian", False)
    if type(hermitian) is not bool:
        raise TypeError("svd hermitian must be a bool")
    full_matrices = attrs.get("full_matrices", True)
    if type(full_matrices) is not bool:
        raise TypeError("svd full_matrices must be a bool")
    return (
        ArraySpec(
            (*batch_shape, rows, rows if full_matrices else core_size),
            input_dtype,
        ),
        ArraySpec((*batch_shape, core_size), result_real_dtype),
        ArraySpec(
            (*batch_shape, columns if full_matrices else core_size, columns),
            input_dtype,
        ),
    )


EVALUATORS: dict[str, ResultEvaluator] = {
    "norm": _norm,
    "vector_norm": _vector_norm,
    "solve": _solve,
    **{
        kind: partial(_matrix_result, kind=kind)
        for kind in (
            "cholesky",
            "det",
            "eigvals",
            "eigvalsh",
            "inv",
            "matrix_norm",
            "pinv",
            "qr_r",
            "svdvals",
        )
    },
    **{
        kind: partial(_decomposition, kind=kind) for kind in ("eig", "eigh", "qr", "slogdet", "svd")
    },
}
