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
    grad_value: Any,
) -> Any:
    hess_blocks_flat = _allocate_hessian_blocks_flat(
        array_ns=context.array_ns,
        primal_flat_sizes=context.primal_flat_sizes,
        primal_dtypes=context.primal_dtypes,
    )
    positions = (
        (col_block, column)
        for col_block, col_size in enumerate(context.primal_flat_sizes)
        for column in range(col_size)
    )
    for position_batch in batched(positions, _PULLBACK_MANY_BATCH_SIZE):
        cotangents = _build_basis_cotangent_batch(
            context=context,
            grad_value=grad_value,
            positions=position_batch,
        )
        hvp_values = linear.transpose_many(cotangents)
        for (col_block, column), hvp_value in zip(
            position_batch,
            hvp_values,
            strict=True,
        ):
            _assign_hessian_column(
                context=context,
                hess_blocks_flat=hess_blocks_flat,
                col_block=col_block,
                column=column,
                hvp_value=hvp_value,
            )

    return _reshape_hessian_blocks(
        hessian_blocks_flat=hess_blocks_flat,
        primal_shapes=context.primal_shapes,
        single_argnum=context.single_argnum,
    )


def _reshape_hessian_diagonals(*, context: _HessianLoopContext, diagonals: list[Any]) -> Any:
    reshaped = tuple(
        diagonal.reshape(shape)
        for diagonal, shape in zip(diagonals, context.primal_shapes, strict=True)
    )
    return reshaped[0] if context.single_argnum else reshaped


def _hessian_diag_reverse_loop(
    *,
    context: _HessianLoopContext,
    linear: LinearMap,
    grad_value: Any,
) -> Any:
    diagonals = [
        context.array_ns.zeros(size, dtype=dtype)
        for size, dtype in zip(
            context.primal_flat_sizes,
            context.primal_dtypes,
            strict=True,
        )
    ]
    positions = (
        (block, column)
        for block, flat_size in enumerate(context.primal_flat_sizes)
        for column in range(flat_size)
    )
    for position_batch in batched(positions, _PULLBACK_MANY_BATCH_SIZE):
        cotangents = _build_basis_cotangent_batch(
            context=context,
            grad_value=grad_value,
            positions=position_batch,
        )
        hvp_values = linear.transpose_many(cotangents)
        for (block, column), hvp_value in zip(
            position_batch,
            hvp_values,
            strict=True,
        ):
            hvp_entries = _normalize_hvp_output(
                hvp_value=hvp_value,
                expected_selected_args=len(context.primal_shapes),
                single_argnum=context.single_argnum,
            )
            diagonals[block][column] = context.array_ns.asarray(hvp_entries[block]).reshape(-1)[
                column
            ]
    return _reshape_hessian_diagonals(context=context, diagonals=diagonals)


def _build_basis_cotangent_batch(
    *,
    context: _HessianLoopContext,
    grad_value: Any,
    positions: tuple[tuple[int, int], ...],
) -> tuple[Any, ...]:
    """Build basis pytrees as row views over one allocation per gradient entry."""
    grad_entries = _normalize_hvp_output(
        hvp_value=grad_value,
        expected_selected_args=len(context.primal_shapes),
        single_argnum=context.single_argnum,
    )
    values_by_seed: list[list[Any]] = [[] for _ in positions]

    for block, entry in enumerate(grad_entries):
        layout = _gradient_entry_layout(context=context, entry=entry, block=block)
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
    entry: Any,
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


def _assign_hessian_column(
    *,
    context: _HessianLoopContext,
    hess_blocks_flat: list[list[Any]],
    col_block: int,
    column: int,
    hvp_value: Any,
) -> None:
    hvp_entries = _normalize_hvp_output(
        hvp_value=hvp_value,
        expected_selected_args=len(context.primal_shapes),
        single_argnum=context.single_argnum,
    )
    for row_block, entry in enumerate(hvp_entries):
        hess_blocks_flat[row_block][col_block][:, column] = context.array_ns.asarray(entry).reshape(
            -1
        )
