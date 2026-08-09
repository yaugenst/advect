"""Explicit array-family transposes and non-differentiability contracts.

Most operations rely on structural transposition of their JVP. The explicit
rules below are the exceptional real-linear or performance-critical adjoints
that earn a direct implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import wrap_array_family_vjp_rule
from advect.autodiff.rules.array_family.vjp.elementwise import (
    _vjp_absolute,
    _vjp_add,
    _vjp_astype,
    _vjp_conjugate,
    _vjp_cos,
    _vjp_divide,
    _vjp_exp,
    _vjp_identity,
    _vjp_imag,
    _vjp_ldexp,
    _vjp_multiply,
    _vjp_negative,
    _vjp_power,
    _vjp_real,
    _vjp_sign,
    _vjp_sin,
    _vjp_subtract,
    _vjp_where,
    _vjp_zero,
)
from advect.autodiff.rules.array_family.vjp.fft import (
    _vjp_fft,
    _vjp_fft2,
    _vjp_fftn,
    _vjp_fftshift,
    _vjp_ifft,
    _vjp_ifft2,
    _vjp_ifftn,
    _vjp_ifftshift,
    _vjp_irfft,
    _vjp_irfft2,
    _vjp_irfftn,
    _vjp_rfft,
    _vjp_rfft2,
    _vjp_rfftn,
)
from advect.autodiff.rules.array_family.vjp.gather import (
    _vjp_bincount,
    _vjp_take,
    _vjp_take_along_axis,
)
from advect.autodiff.rules.array_family.vjp.linalg.contractions import (
    _vjp_dot,
    _vjp_einsum,
    _vjp_matmul,
    _vjp_tensordot,
    _vjp_vecdot,
)
from advect.autodiff.rules.array_family.vjp.linalg.decompositions import (
    _vjp_cholesky,
    _vjp_pinv,
    _vjp_qr,
    _vjp_svd,
    _vjp_svdvals,
)
from advect.autodiff.rules.array_family.vjp.linalg.eigen import (
    _vjp_eigh,
    _vjp_eigvalsh,
)
from advect.autodiff.rules.array_family.vjp.linear import (
    _vjp_atleast,
    _vjp_concatenate,
    _vjp_cross,
    _vjp_cumsum,
    _vjp_diag,
    _vjp_diagonal,
    _vjp_diff,
    _vjp_flip,
    _vjp_fliplr,
    _vjp_flipud,
    _vjp_gradient,
    _vjp_inner,
    _vjp_kron,
    _vjp_linspace,
    _vjp_outer,
    _vjp_pad,
    _vjp_ravel,
    _vjp_repeat,
    _vjp_roll,
    _vjp_rollaxis,
    _vjp_rot90,
    _vjp_solve,
    _vjp_stack,
    _vjp_swapaxes,
    _vjp_tile,
    _vjp_trace,
    _vjp_tril,
    _vjp_triu,
)
from advect.autodiff.rules.array_family.vjp.reductions_indexing import (
    _vjp_getitem,
    _vjp_index_update,
    _vjp_mean,
    _vjp_sum,
)
from advect.autodiff.rules.array_family.vjp.shape_creation import (
    _vjp_broadcast_to,
    _vjp_expand_dims,
    _vjp_moveaxis,
    _vjp_ones_like,
    _vjp_reshape,
    _vjp_squeeze,
    _vjp_transpose,
    _vjp_zeros_like,
)
from advect.autodiff.rules.array_family.vjp.signal import (
    _vjp_convolve,
    _vjp_correlate,
)

__all__ = [
    "non_differentiable_items",
    "vjp_rule_items",
]

_VJPFn = Callable[..., tuple[Any | None, ...]]

_EXCEPTION_VJP_REGISTRATIONS: tuple[tuple[str, _VJPFn, bool], ...] = (
    ("array.add", _vjp_add, False),
    ("array.subtract", _vjp_subtract, False),
    ("array.negative", _vjp_negative, False),
    ("array.positive", _vjp_identity, False),
    ("array.conjugate", _vjp_conjugate, False),
    ("array.astype", _vjp_astype, True),
    ("array.real", _vjp_real, True),
    ("array.where", _vjp_where, True),
    ("advect.copy", _vjp_identity, False),
    ("array.multiply", _vjp_multiply, True),
    ("array.divide", _vjp_divide, True),
    ("array_ext.true_divide", _vjp_divide, True),
    ("array.power", _vjp_power, True),
    ("array.sin", _vjp_sin, True),
    ("array.cos", _vjp_cos, True),
    ("array.exp", _vjp_exp, False),
    ("array_ext.ldexp", _vjp_ldexp, True),
    ("array.sign", _vjp_sign, True),
    ("array.floor", _vjp_zero, True),
    ("array.ceil", _vjp_zero, True),
    ("array.trunc", _vjp_zero, True),
    ("array.rint", _vjp_zero, True),
    ("array_ext.spacing", _vjp_zero, True),
    ("array.floor_divide", _vjp_zero, True),
    ("array.reshape", _vjp_reshape, True),
    ("array.transpose", _vjp_transpose, False),
    ("array.moveaxis", _vjp_moveaxis, False),
    ("array.squeeze", _vjp_squeeze, True),
    ("array.expand_dims", _vjp_expand_dims, True),
    ("array.broadcast_to", _vjp_broadcast_to, True),
    ("array.zeros_like", _vjp_zeros_like, False),
    ("array.ones_like", _vjp_ones_like, False),
    ("array.sum", _vjp_sum, True),
    ("array.mean", _vjp_mean, True),
    ("advect.getitem", _vjp_getitem, True),
    ("advect.index_update", _vjp_index_update, False),
    ("array_ext.bincount", _vjp_bincount, True),
    ("array.take", _vjp_take, True),
    ("array.take_along_axis", _vjp_take_along_axis, True),
    ("array.absolute", _vjp_absolute, True),
    ("array.imag", _vjp_imag, True),
    ("array.matmul", _vjp_matmul, True),
    ("array_ext.dot", _vjp_dot, True),
    ("array.tensordot", _vjp_tensordot, True),
    ("array_ext.einsum", _vjp_einsum, True),
    ("array_ext.convolve", _vjp_convolve, True),
    ("array_ext.correlate", _vjp_correlate, True),
    ("array.concatenate", _vjp_concatenate, True),
    ("array.stack", _vjp_stack, True),
    ("array_ext.ravel", _vjp_ravel, True),
    ("array.swapaxes", _vjp_swapaxes, False),
    ("array.flip", _vjp_flip, False),
    ("array_ext.fliplr", _vjp_fliplr, False),
    ("array_ext.flipud", _vjp_flipud, False),
    ("array.roll", _vjp_roll, False),
    ("array_ext.rot90", _vjp_rot90, False),
    ("array_ext.rollaxis", _vjp_rollaxis, False),
    ("array.triu", _vjp_triu, False),
    ("array.tril", _vjp_tril, False),
    ("array.atleast_1d", _vjp_atleast, True),
    ("array.atleast_2d", _vjp_atleast, True),
    ("array.atleast_3d", _vjp_atleast, True),
    ("array_ext.diag", _vjp_diag, False),
    ("array.diagonal", _vjp_diagonal, True),
    ("array.trace", _vjp_trace, True),
    ("array.cumsum", _vjp_cumsum, True),
    ("array_ext.pad", _vjp_pad, True),
    ("array.diff", _vjp_diff, True),
    ("array.repeat", _vjp_repeat, True),
    ("array.tile", _vjp_tile, True),
    ("array_ext.gradient", _vjp_gradient, True),
    ("array_ext.inner", _vjp_inner, True),
    ("array.outer", _vjp_outer, True),
    ("array.cross", _vjp_cross, True),
    ("array_ext.kron", _vjp_kron, True),
    ("array.linspace", _vjp_linspace, False),
    ("array_ext.linalg.solve", _vjp_solve, True),
    ("array_ext.linalg.cholesky", _vjp_cholesky, True),
    ("array_ext.linalg.eigh", _vjp_eigh, True),
    ("array_ext.linalg.eigvalsh", _vjp_eigvalsh, True),
    ("array_ext.linalg.pinv", _vjp_pinv, True),
    ("array_ext.linalg.qr", _vjp_qr, True),
    ("array_ext.linalg.svd", _vjp_svd, True),
    ("array_ext.linalg.svdvals", _vjp_svdvals, True),
    ("array.vecdot", _vjp_vecdot, True),
    ("array_ext.fft.fft", _vjp_fft, True),
    ("array_ext.fft.ifft", _vjp_ifft, True),
    ("array_ext.fft.fft2", _vjp_fft2, True),
    ("array_ext.fft.ifft2", _vjp_ifft2, True),
    ("array_ext.fft.fftn", _vjp_fftn, True),
    ("array_ext.fft.ifftn", _vjp_ifftn, True),
    ("array_ext.fft.rfft", _vjp_rfft, True),
    ("array_ext.fft.rfft2", _vjp_rfft2, True),
    ("array_ext.fft.rfftn", _vjp_rfftn, True),
    ("array_ext.fft.irfft", _vjp_irfft, True),
    ("array_ext.fft.irfft2", _vjp_irfft2, True),
    ("array_ext.fft.irfftn", _vjp_irfftn, True),
    ("array_ext.fft.fftshift", _vjp_fftshift, False),
    ("array_ext.fft.ifftshift", _vjp_ifftshift, False),
)

_EXCEPTION_VJP_NEEDS_OUTPUT_OPS: frozenset[str] = frozenset(
    {
        "array.power",
        "array.exp",
        "array.matmul",
        "array_ext.dot",
        "array.tensordot",
        "array_ext.einsum",
        "array_ext.linalg.solve",
        "array_ext.linalg.cholesky",
        "array_ext.linalg.pinv",
        "array_ext.linalg.qr",
        "array_ext.linalg.svd",
        "array_ext.linalg.eigh",
        "array.vecdot",
    }
)

_NON_DIFFERENTIABLE_OP_CONTRACTS = (
    (
        "array.all",
        "Boolean reductions are not differentiable.",
    ),
    (
        "array.any",
        "Boolean reductions are not differentiable.",
    ),
    (
        "array.argmin",
        "Arg reductions return integer indices and are not differentiable.",
    ),
    (
        "array.argmax",
        "Arg reductions return integer indices and are not differentiable.",
    ),
    (
        "array.argsort",
        "Sorting indices are discrete and are not differentiable.",
    ),
    (
        "array.count_nonzero",
        "Count reductions return integer values and are not differentiable.",
    ),
    (
        "array.searchsorted",
        "Insertion indices are discrete and are not differentiable.",
    ),
    (
        "array.equal",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.not_equal",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.less",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.less_equal",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.greater",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.greater_equal",
        "Comparison operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.isnan",
        "NaN predicate operations produce boolean outputs and are not differentiable.",
    ),
    (
        "array.isfinite",
        "Finite-value predicates produce boolean outputs and are not differentiable.",
    ),
    (
        "array.isinf",
        "Infinity predicates produce boolean outputs and are not differentiable.",
    ),
    (
        "array.signbit",
        "Sign-bit predicates produce boolean outputs and are not differentiable.",
    ),
    (
        "array.invert",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.logical_not",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.logical_and",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.logical_or",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.logical_xor",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.bitwise_and",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.bitwise_or",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
    (
        "array.bitwise_xor",
        "Boolean/discrete masking operations produce non-differentiable outputs.",
    ),
)

_DYNAMIC_ONLY_NON_DIFFERENTIABLE_OPS = (
    "array.unique_counts",
    "array.unique_inverse",
    "array_ext.unique",
    "array_ext.unique_index",
    "array_ext.unique_index_counts",
    "array_ext.unique_index_inverse",
    "array_ext.unique_index_inverse_counts",
    "array_ext.unique_inverse_counts",
)


def vjp_rule_items() -> tuple[tuple[str, _VJPFn, bool, bool], ...]:
    """Build canonical VJP payloads for the built-in operation definitions."""
    return tuple(
        (
            name,
            wrap_array_family_vjp_rule(rule),
            needs_inputs,
            name in _EXCEPTION_VJP_NEEDS_OUTPUT_OPS,
        )
        for name, rule, needs_inputs in _EXCEPTION_VJP_REGISTRATIONS
    )


def non_differentiable_items() -> tuple[tuple[str, str], ...]:
    """Return explicit non-differentiability contracts for built-in operations."""
    reason = "Unique/set operations are discrete and not differentiable."
    return (
        *_NON_DIFFERENTIABLE_OP_CONTRACTS,
        *((name, reason) for name in _DYNAMIC_ONLY_NON_DIFFERENTIABLE_OPS),
    )
