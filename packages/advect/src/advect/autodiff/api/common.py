"""Shared helpers for public autodiff APIs."""

from __future__ import annotations

import math
from importlib import import_module
from typing import TYPE_CHECKING, NoReturn

from advect.autodiff.api.inputs import _normalize_argnums_for_call
from advect.autodiff.rules.array_family.providers import resolve_array_family_backend_provider
from advect.core._errors import HigherOrderNotSupportedError

if TYPE_CHECKING:
    from typing import Any

_HIGHER_ORDER_NAMESPACE_ATTRS = (
    "asarray",
    "result_type",
    "zeros",
    "zeros_like",
    "diag",
    "float64",
)


def _require_array_namespace_for_higher_order(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    scalar_fallback_values: tuple[Any, ...] | None = None,
) -> Any:
    values: tuple[Any, ...] = args + tuple(kwargs.values())
    try:
        provider = resolve_array_family_backend_provider(*values)
    except RuntimeError as exc:
        fallback_values = values if scalar_fallback_values is None else scalar_fallback_values
        if fallback_values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in fallback_values
        ):
            import numpy as np  # noqa: PLC0415 - scalar Hessians use the default provider

            import_module("advect.numpy")
            return np
        msg = (
            "Higher-order autodiff APIs require a runtime array namespace for dense "
            "Hessian assembly. Pass arrays implementing __array_namespace__."
        )
        raise HigherOrderNotSupportedError(msg) from exc

    namespace = provider.namespace
    missing = [
        attr_name
        for attr_name in _HIGHER_ORDER_NAMESPACE_ATTRS
        if not hasattr(namespace, attr_name)
    ]
    if missing:
        missing_joined = ", ".join(sorted(missing))
        msg = (
            f"Higher-order autodiff requires backend namespace capabilities "
            f"{_HIGHER_ORDER_NAMESPACE_ATTRS}, but backend '{provider.backend}' is missing: "
            f"{missing_joined}."
        )
        raise HigherOrderNotSupportedError(msg)
    return namespace


def _resolve_selected_argnums(
    *,
    argnums: tuple[int, ...],
    nargs: int,
) -> list[int]:
    resolved = _normalize_argnums_for_call(argnums, nargs=nargs)
    if not resolved:
        msg = "Higher-order APIs require at least one selected argument."
        raise ValueError(msg)
    return resolved


def _normalize_hvp_output(
    *,
    hvp_value: Any,
    expected_selected_args: int,
    single_argnum: bool,
) -> tuple[Any, ...]:
    if single_argnum:
        return (hvp_value,)
    if isinstance(hvp_value, tuple):
        if len(hvp_value) != expected_selected_args:
            msg = (
                "Higher-order API returned an unexpected Hessian-vector-product output shape. "
                f"Expected {expected_selected_args} entries, got {len(hvp_value)}."
            )
            raise HigherOrderNotSupportedError(msg)
        return hvp_value
    if expected_selected_args == 1:
        return (hvp_value,)
    msg = (
        "Higher-order API returned an unexpected Hessian-vector-product output type. "
        f"Expected tuple for {expected_selected_args} selected arguments."
    )
    raise HigherOrderNotSupportedError(msg)


def _prepare_higher_order_inputs(
    *,
    array_ns: Any,
    args: tuple[Any, ...],
    argnums: tuple[int, ...],
) -> tuple[list[tuple[int, ...]], list[int], list[Any]]:
    resolved_argnums = _resolve_selected_argnums(argnums=argnums, nargs=len(args))
    selected_values = [args[arg_index] for arg_index in resolved_argnums]
    primal_arrays = [array_ns.asarray(primal) for primal in selected_values]
    primal_shapes = [tuple(int(d) for d in primal.shape) for primal in primal_arrays]
    primal_flat_sizes = [math.prod(shape) for shape in primal_shapes]
    primal_dtypes = [
        array_ns.result_type(primal.dtype, array_ns.float64) for primal in primal_arrays
    ]
    return primal_shapes, primal_flat_sizes, primal_dtypes


def _allocate_hessian_blocks_flat(
    *,
    array_ns: Any,
    primal_flat_sizes: list[int],
    primal_dtypes: list[Any],
) -> list[list[Any]]:
    n_selected = len(primal_flat_sizes)
    return [
        [
            array_ns.zeros(
                (primal_flat_sizes[row], primal_flat_sizes[col]),
                dtype=array_ns.result_type(primal_dtypes[row], primal_dtypes[col]),
            )
            for col in range(n_selected)
        ]
        for row in range(n_selected)
    ]


def _reshape_hessian_blocks(
    *,
    hessian_blocks_flat: list[list[Any]],
    primal_shapes: list[tuple[int, ...]],
    single_argnum: bool,
) -> Any:
    hess_blocks = tuple(
        tuple(
            hessian_blocks_flat[row][col].reshape(primal_shapes[row] + primal_shapes[col])
            for col in range(len(primal_shapes))
        )
        for row in range(len(primal_shapes))
    )
    if single_argnum:
        return hess_blocks[0][0]
    return hess_blocks


def _raise_hessian_gradient_structure_error() -> NoReturn:
    msg = "Higher-order API returned an unexpected gradient structure for Hessian construction."
    raise HigherOrderNotSupportedError(msg)
