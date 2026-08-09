"""JAX reverse-mode bridge for pure NumPy-backed Advect callables."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

import advect.autodiff as ad
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.interop._common import (
    conjugate_complex_tree,
    differentiable_argnums,
    numeric_tree,
    require_dependency,
)

if TYPE_CHECKING:
    from collections.abc import Callable

jax = require_dependency("jax")


def _split_aux(value: Any, *, has_aux: bool, boundary: str) -> tuple[Any, Any | None]:
    if not has_aux:
        return value, None
    match value:
        case (output, aux) if isinstance(value, tuple):
            return output, aux
    message = f"{boundary} must be a (value, aux) tuple when has_aux=True"
    raise TypeError(message)


def _validate_specs(result_shape_dtypes: Any, *, has_aux: bool) -> Any:
    value_specs, _aux_specs = _split_aux(
        result_shape_dtypes,
        has_aux=has_aux,
        boundary="JAX result_shape_dtypes",
    )
    leaves = jax.tree_util.tree_leaves(value_specs)
    if not leaves:
        message = "JAX result_shape_dtypes value must contain at least one output leaf"
        raise TypeError(message)
    for index, spec in enumerate(leaves):
        if not hasattr(spec, "shape") or not hasattr(spec, "dtype"):
            message = f"JAX result_shape_dtypes value leaf {index} must expose shape and dtype"
            raise TypeError(message)
        dtype = np.dtype(spec.dtype)
        if dtype.kind not in {"f", "c"}:
            message = (
                f"JAX result_shape_dtypes value leaf {index} has dtype {dtype}; "
                "only NumPy floating and complex outputs are supported"
            )
            raise TypeError(message)
    return jax.tree_util.tree_structure(result_shape_dtypes)


def _output_payload(value: Any, expected_treedef: Any | None, *, has_aux: bool) -> Any:
    output, _aux = _split_aux(value, has_aux=has_aux, boundary="Advect function output")
    numeric_tree(output, boundary="JAX bridge output")
    leaves, treedef = jax.tree_util.tree_flatten(value)
    if expected_treedef is not None and treedef != expected_treedef:
        message = "Advect output pytree does not match JAX result_shape_dtypes"
        raise TypeError(message)
    return jax.tree_util.tree_unflatten(
        treedef,
        [np.asarray(leaf) for leaf in leaves],
    )


def _validate_inputs(values: tuple[Any, ...]) -> None:
    differentiable_argnums(len(values))
    leaves, _treedef = jax.tree_util.tree_flatten(values)
    if not leaves:
        message = "JAX bridge inputs must contain at least one array leaf"
        raise TypeError(message)
    for index, leaf in enumerate(leaves):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            message = f"JAX bridge input leaf {index} is not an array"
            raise TypeError(message)
        dtype = np.dtype(leaf.dtype)
        if dtype.kind not in {"f", "c"}:
            message = (
                f"JAX bridge input leaf {index} has dtype {dtype}; "
                "only NumPy floating and complex arrays are supported"
            )
            raise TypeError(message)


def _numpy_tree(value: Any) -> Any:
    return jax.tree_util.tree_map(np.asarray, value)


def _require_result_specs_when_staged(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        message = (
            "JAX result_shape_dtypes is required when an Advect bridge is staged; "
            "eager execution and reverse mode work without it. Pass "
            "result_shape_dtypes=... to advect.interop.jax.wrap for jax.jit or "
            "jax.eval_shape."
        )
        raise TypeError(message)


def _gradient_payloads(gradients: Any, primals: tuple[Any, ...]) -> tuple[Any, ...]:
    gradients = conjugate_complex_tree(gradients)
    gradient_leaves, _gradient_treedef = tree_flatten(gradients)
    primal_leaves, primal_treedef = tree_flatten(primals)
    payloads: list[np.ndarray] = []
    for gradient, primal in zip(gradient_leaves, primal_leaves, strict=True):
        primal_array = np.asarray(primal)
        array = np.asarray(gradient)
        if primal_array.dtype.kind != "c":
            array = np.real(array)
        payloads.append(np.asarray(array, dtype=primal_array.dtype))
    return tree_unflatten(primal_treedef, payloads)


def wrap(
    function: Callable[..., Any],
    *,
    has_aux: bool = False,
    result_shape_dtypes: Any | None = None,
) -> Callable[..., Any]:
    """Wrap a pure NumPy-backed callable as a first-order JAX operation.

    With ``has_aux=True``, ``function`` returns ``(value, aux)`` and only
    ``value`` participates in the Advect VJP. Eager calls infer their outputs
    by executing ``function`` directly.
    JIT compilation and abstract shape evaluation require
    ``result_shape_dtypes``, a JAX pytree of objects with ``shape`` and
    ``dtype`` attributes, normally
    ``jax.ShapeDtypeStruct`` objects. Reverse mode replays ``function`` to
    build and consume an Advect pullback, so it must be pure and deterministic.
    """
    result_treedef = (
        None
        if result_shape_dtypes is None
        else _validate_specs(result_shape_dtypes, has_aux=has_aux)
    )

    def differentiable_function(*values: Any) -> Any:
        value, _aux = _split_aux(
            function(*values),
            has_aux=has_aux,
            boundary="Advect function output",
        )
        return value

    def forward_callback(*values: Any) -> Any:
        concrete_values = _numpy_tree(values)
        return _output_payload(
            function(*concrete_values),
            result_treedef,
            has_aux=has_aux,
        )

    def backward_callback(values: tuple[Any, ...], cotangents: Any) -> tuple[Any, ...]:
        concrete_values = _numpy_tree(values)
        value_cotangents, _aux_cotangents = _split_aux(
            cotangents,
            has_aux=has_aux,
            boundary="JAX output cotangents",
        )
        concrete_cotangents = _numpy_tree(value_cotangents)
        _, pullback = ad.vjp(
            differentiable_function,
            argnums=differentiable_argnums(len(concrete_values)),
        )(*concrete_values)
        gradients = pullback(conjugate_complex_tree(concrete_cotangents))
        return _gradient_payloads(gradients, concrete_values)

    def call(*values: Any) -> Any:
        _validate_inputs(values)
        if result_shape_dtypes is None:
            _require_result_specs_when_staged(values)
            return jax.device_put(forward_callback(*values))
        return jax.pure_callback(
            forward_callback,
            result_shape_dtypes,
            *values,
        )

    @jax.custom_vjp
    def primitive(*values: Any) -> Any:
        return call(*values)

    def forward_rule(*values: Any) -> tuple[Any, tuple[Any, ...]]:
        return call(*values), values

    def backward_rule(values: tuple[Any, ...], cotangents: Any) -> tuple[Any, ...]:
        if result_shape_dtypes is None:
            _require_result_specs_when_staged((values, cotangents))
            return tuple(jax.device_put(backward_callback(values, cotangents)))
        input_specs = jax.tree_util.tree_map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
            values,
        )
        gradients = jax.pure_callback(
            backward_callback,
            input_specs,
            values,
            cotangents,
        )
        return tuple(gradients)

    primitive.defvjp(forward_rule, backward_rule)
    return functools.wraps(function)(primitive)


__all__ = ["wrap"]
