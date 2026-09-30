"""Explicit array-family transposes and non-differentiability contracts.

Diagonal elementwise operations transpose their shared JVP partials in closed
form, which is measured to be an order of magnitude faster than tracing and
transposing their JVPs. Other operations rely on structural transposition of
their JVP; the explicit rules below are the exceptional real-linear or
performance-critical adjoints that earn a direct implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from advect.autodiff.rules.array_family.jvp.elementwise_partials import ELEMENTWISE_PARTIALS
from advect.autodiff.rules.array_family.vjp.elementwise import (
    _vjp_absolute,
    _vjp_astype,
    _vjp_conjugate,
    _vjp_imag,
    _vjp_ldexp,
    _vjp_real,
    _vjp_sign,
    _vjp_where,
    make_diagonal_vjp_from_partials,
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
    _vjp_scatter_add,
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

_VJPFn = Callable[..., tuple[Any | None, ...]]

_EXCEPTION_VJP_REGISTRATIONS: tuple[tuple[str, _VJPFn, bool, bool], ...] = (
    ("array.conjugate", _vjp_conjugate, False, False),
    ("array.astype", _vjp_astype, True, False),
    ("array.real", _vjp_real, False, False),
    ("array.where", _vjp_where, True, False),
    ("array_ext.ldexp", _vjp_ldexp, True, False),
    ("array.sign", _vjp_sign, True, False),
    ("array.reshape", _vjp_reshape, True, False),
    ("array.transpose", _vjp_transpose, False, False),
    ("array.moveaxis", _vjp_moveaxis, False, False),
    ("array.squeeze", _vjp_squeeze, True, False),
    ("array.expand_dims", _vjp_expand_dims, True, False),
    ("array.broadcast_to", _vjp_broadcast_to, True, False),
    ("array.zeros_like", _vjp_zeros_like, False, False),
    ("array.ones_like", _vjp_ones_like, False, False),
    ("array.sum", _vjp_sum, True, False),
    ("array.mean", _vjp_mean, True, False),
    ("advect.getitem", _vjp_getitem, True, False),
    ("advect.index_update", _vjp_index_update, False, False),
    ("array_ext.bincount", _vjp_bincount, True, False),
    ("array.take", _vjp_take, True, False),
    ("array.take_along_axis", _vjp_take_along_axis, True, False),
    ("advect.scatter_add", _vjp_scatter_add, True, False),
    ("array.absolute", _vjp_absolute, True, False),
    ("array.imag", _vjp_imag, True, False),
    ("array.matmul", _vjp_matmul, True, False),
    ("array_ext.dot", _vjp_dot, True, False),
    ("array.tensordot", _vjp_tensordot, True, False),
    ("array_ext.einsum", _vjp_einsum, True, False),
    ("array_ext.convolve", _vjp_convolve, True, False),
    ("array_ext.correlate", _vjp_correlate, True, False),
    ("array.concatenate", _vjp_concatenate, True, False),
    ("array.stack", _vjp_stack, True, False),
    ("array_ext.ravel", _vjp_ravel, True, False),
    ("array.swapaxes", _vjp_swapaxes, False, False),
    ("array.flip", _vjp_flip, False, False),
    ("array_ext.fliplr", _vjp_fliplr, False, False),
    ("array_ext.flipud", _vjp_flipud, False, False),
    ("array.roll", _vjp_roll, False, False),
    ("array_ext.rot90", _vjp_rot90, False, False),
    ("array_ext.rollaxis", _vjp_rollaxis, False, False),
    ("array.triu", _vjp_triu, False, False),
    ("array.tril", _vjp_tril, False, False),
    ("array.atleast_1d", _vjp_atleast, True, False),
    ("array.atleast_2d", _vjp_atleast, True, False),
    ("array.atleast_3d", _vjp_atleast, True, False),
    ("array_ext.diag", _vjp_diag, True, False),
    ("array.diagonal", _vjp_diagonal, True, False),
    ("array.trace", _vjp_trace, True, False),
    ("array.cumsum", _vjp_cumsum, True, False),
    ("array_ext.pad", _vjp_pad, True, False),
    ("array.diff", _vjp_diff, True, False),
    ("array.repeat", _vjp_repeat, True, False),
    ("array.tile", _vjp_tile, True, False),
    ("array_ext.gradient", _vjp_gradient, True, False),
    ("array_ext.inner", _vjp_inner, True, False),
    ("array.outer", _vjp_outer, True, False),
    ("array.cross", _vjp_cross, True, False),
    ("array_ext.kron", _vjp_kron, True, False),
    ("array.linspace", _vjp_linspace, False, False),
    ("array_ext.linalg.solve", _vjp_solve, True, True),
    ("array_ext.linalg.cholesky", _vjp_cholesky, True, True),
    ("array_ext.linalg.eigh", _vjp_eigh, True, True),
    ("array_ext.linalg.eigvalsh", _vjp_eigvalsh, True, False),
    ("array_ext.linalg.pinv", _vjp_pinv, True, True),
    ("array_ext.linalg.qr", _vjp_qr, True, True),
    ("array_ext.linalg.svd", _vjp_svd, True, True),
    ("array_ext.linalg.svdvals", _vjp_svdvals, True, False),
    ("array.vecdot", _vjp_vecdot, True, False),
    ("array_ext.fft.fft", _vjp_fft, True, False),
    ("array_ext.fft.ifft", _vjp_ifft, True, False),
    ("array_ext.fft.fft2", _vjp_fft2, True, False),
    ("array_ext.fft.ifft2", _vjp_ifft2, True, False),
    ("array_ext.fft.fftn", _vjp_fftn, True, False),
    ("array_ext.fft.ifftn", _vjp_ifftn, True, False),
    ("array_ext.fft.rfft", _vjp_rfft, True, False),
    ("array_ext.fft.rfft2", _vjp_rfft2, True, False),
    ("array_ext.fft.rfftn", _vjp_rfftn, True, False),
    ("array_ext.fft.irfft", _vjp_irfft, True, False),
    ("array_ext.fft.irfft2", _vjp_irfft2, True, False),
    ("array_ext.fft.irfftn", _vjp_irfftn, True, False),
    ("array_ext.fft.fftshift", _vjp_fftshift, False, False),
    ("array_ext.fft.ifftshift", _vjp_ifftshift, False, False),
)

#: Each reason with the operations it makes non-differentiable.
_NON_DIFFERENTIABLE_OPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Boolean reductions are not differentiable.", ("array.all", "array.any")),
    (
        "Arg reductions return integer indices and are not differentiable.",
        ("array.argmin", "array.argmax"),
    ),
    ("Sorting indices are discrete and are not differentiable.", ("array.argsort",)),
    (
        "Count reductions return integer values and are not differentiable.",
        ("array.count_nonzero",),
    ),
    ("Insertion indices are discrete and are not differentiable.", ("array.searchsorted",)),
    (
        "Comparison operations produce boolean outputs and are not differentiable.",
        (
            "array.equal",
            "array.not_equal",
            "array.less",
            "array.less_equal",
            "array.greater",
            "array.greater_equal",
        ),
    ),
    (
        "NaN predicate operations produce boolean outputs and are not differentiable.",
        ("array.isnan",),
    ),
    (
        "Finite-value predicates produce boolean outputs and are not differentiable.",
        ("array.isfinite",),
    ),
    (
        "Infinity predicates produce boolean outputs and are not differentiable.",
        ("array.isinf",),
    ),
    (
        "Sign-bit predicates produce boolean outputs and are not differentiable.",
        ("array.signbit",),
    ),
    (
        "Boolean/discrete masking operations produce non-differentiable outputs.",
        (
            "array.invert",
            "array.logical_not",
            "array.logical_and",
            "array.logical_or",
            "array.logical_xor",
            "array.bitwise_and",
            "array.bitwise_or",
            "array.bitwise_xor",
        ),
    ),
    (
        "Unique/set operations are discrete and not differentiable.",
        (
            "array.unique_counts",
            "array.unique_inverse",
            "array_ext.unique",
            "array_ext.unique_index",
            "array_ext.unique_index_counts",
            "array_ext.unique_index_inverse",
            "array_ext.unique_index_inverse_counts",
            "array_ext.unique_inverse_counts",
        ),
    ),
)


def vjp_rule_items() -> tuple[tuple[str, _VJPFn, bool, bool], ...]:
    """Build canonical VJP payloads for the built-in operation definitions."""
    items = (
        *_EXCEPTION_VJP_REGISTRATIONS,
        *(
            (
                name,
                make_diagonal_vjp_from_partials(partials),
                partials.reads_operands,
                partials.reads_result,
            )
            for name, partials in ELEMENTWISE_PARTIALS.items()
        ),
    )
    names = tuple(name for name, *_rest in items)
    if len(names) != len(set(names)):
        msg = "Duplicate array-family VJP declaration"
        raise RuntimeError(msg)
    return items


def non_differentiable_items() -> tuple[tuple[str, str], ...]:
    """Return explicit non-differentiability contracts for built-in operations."""
    items = tuple((name, reason) for reason, names in _NON_DIFFERENTIABLE_OPS for name in names)
    names = tuple(name for name, _reason in items)
    if len(names) != len(set(names)):
        msg = "Duplicate array-family non-differentiability declaration"
        raise RuntimeError(msg)
    return items
