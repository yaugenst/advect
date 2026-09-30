"""Dense Hessian assembly loops shared by higher-order APIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any

from advect.autodiff._ephemeral import _PULLBACK_MANY_BATCH_SIZE
from advect.autodiff.api.common import (
    _allocate_hessian_blocks_flat,
    _normalize_hvp_output,
    _raise_hessian_gradient_structure_error,
    _reshape_hessian_blocks,
)
from advect.core._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Iterator

    from advect.autodiff._ephemeral import LinearMap


@dataclass(frozen=True, slots=True)
class _HessianLoopContext:
    array_ns: Any
    primal_shapes: list[tuple[int, ...]]
    primal_flat_sizes: list[int]
    primal_dtypes: list[Any]
    single_argnum: bool


@dataclass(frozen=True, slots=True)
class _GradientEntryLayout:
    treedef: Any
    leaf_shapes: tuple[tuple[int, ...], ...]
    leaf_sizes: tuple[int, ...]


def _hessian_reverse_loop(
    *,
    context: _HessianLoopContext,
    linear: LinearMap,
    grad_value: object,
) -> object:
    hess_blocks_flat = _allocate_hessian_blocks_flat(
        array_ns=context.array_ns,
        primal_flat_sizes=context.primal_flat_sizes,
        primal_dtypes=context.primal_dtypes,
    )
    for col_block, column, hvp_entries in _basis_hvp_columns(
        context=context,
        linear=linear,
        grad_value=grad_value,
    ):
        for row_block, entry in enumerate(hvp_entries):
            hess_blocks_flat[row_block][col_block][:, column] = context.array_ns.asarray(
                entry
            ).reshape(-1)

    return _reshape_hessian_blocks(
        hessian_blocks_flat=hess_blocks_flat,
        primal_shapes=context.primal_shapes,
        single_argnum=context.single_argnum,
    )


def _hessian_diag_reverse_loop(
    *,
    context: _HessianLoopContext,
    linear: LinearMap,
    grad_value: object,
) -> object:
    diagonals = [
        context.array_ns.zeros(size, dtype=dtype)
        for size, dtype in zip(
            context.primal_flat_sizes,
            context.primal_dtypes,
            strict=True,
        )
    ]
    for block, column, hvp_entries in _basis_hvp_columns(
        context=context,
        linear=linear,
        grad_value=grad_value,
    ):
        diagonals[block][column] = context.array_ns.asarray(hvp_entries[block]).reshape(-1)[column]
    reshaped = tuple(
        diagonal.reshape(shape)
        for diagonal, shape in zip(diagonals, context.primal_shapes, strict=True)
    )
    return reshaped[0] if context.single_argnum else reshaped


def _basis_hvp_columns(
    *,
    context: _HessianLoopContext,
    linear: LinearMap,
    grad_value: object,
) -> Iterator[tuple[int, int, tuple[Any, ...]]]:
    """Yield ``(block, column, hvp_entries)`` for each selected input coordinate."""
    layouts = tuple(
        _gradient_entry_layout(context=context, entry=entry, block=block)
        for block, entry in enumerate(_hvp_entries(context, grad_value))
    )
    positions = (
        (block, column)
        for block, flat_size in enumerate(context.primal_flat_sizes)
        for column in range(flat_size)
    )
    for position_batch in batched(positions, _PULLBACK_MANY_BATCH_SIZE):
        cotangents = _build_basis_cotangent_batch(
            context=context,
            layouts=layouts,
            positions=position_batch,
        )
        for (block, column), hvp_value in zip(
            position_batch,
            linear.transpose_many(cotangents),
            strict=True,
        ):
            yield block, column, _hvp_entries(context, hvp_value)


def _hvp_entries(context: _HessianLoopContext, value: object) -> tuple[Any, ...]:
    return _normalize_hvp_output(
        hvp_value=value,
        expected_selected_args=len(context.primal_shapes),
        single_argnum=context.single_argnum,
    )


def _build_basis_cotangent_batch(
    *,
    context: _HessianLoopContext,
    layouts: tuple[_GradientEntryLayout, ...],
    positions: tuple[tuple[int, int], ...],
) -> tuple[object, ...]:
    """Build basis pytrees as row views over one allocation per gradient entry."""
    values_by_seed: list[list[Any]] = [[] for _ in positions]

    for block, layout in enumerate(layouts):
        rows = context.array_ns.zeros(
            (len(positions), context.primal_flat_sizes[block]),
            dtype=context.primal_dtypes[block],
        )

        for seed, (active_block, column) in enumerate(positions):
            if active_block == block:
                rows[seed, column] = 1.0

        for seed in range(len(positions)):
            offset = 0
            leaves: list[Any] = []
            for shape, size in zip(layout.leaf_shapes, layout.leaf_sizes, strict=True):
                leaves.append(rows[seed, offset : offset + size].reshape(shape))
                offset += size
            values_by_seed[seed].append(tree_unflatten(layout.treedef, leaves))

    if context.single_argnum:
        return tuple(values[0] for values in values_by_seed)
    return tuple(tuple(values) for values in values_by_seed)


def _gradient_entry_layout(
    *,
    context: _HessianLoopContext,
    entry: object,
    block: int,
) -> _GradientEntryLayout:
    """Validate and flatten one dense array or registered array-container gradient."""
    if not hasattr(entry, "shape") or not hasattr(entry, "dtype"):
        _raise_hessian_gradient_structure_error()
    entry_arr = context.array_ns.asarray(entry)
    if tuple(int(dimension) for dimension in entry_arr.shape) != context.primal_shapes[block]:
        _raise_hessian_gradient_structure_error()

    leaves, treedef = tree_flatten(entry)
    if not leaves:
        _raise_hessian_gradient_structure_error()

    leaf_arrays = tuple(context.array_ns.asarray(leaf) for leaf in leaves)
    leaf_shapes = tuple(tuple(int(dimension) for dimension in leaf.shape) for leaf in leaf_arrays)
    leaf_sizes = tuple(
        math.prod(int(dimension) for dimension in leaf.shape) for leaf in leaf_arrays
    )
    if sum(leaf_sizes) != context.primal_flat_sizes[block]:
        _raise_hessian_gradient_structure_error()
    return _GradientEntryLayout(
        treedef=treedef,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
    )
