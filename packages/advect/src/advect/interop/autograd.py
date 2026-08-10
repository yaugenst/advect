"""HIPS Autograd reverse-mode bridge for Advect callables."""

from __future__ import annotations

import functools
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._pytree import tree_flatten, tree_unflatten
from advect.interop._common import (
    conjugate_complex_tree,
    numeric_tree,
    require_dependency,
    validated_vjp,
)

if TYPE_CHECKING:
    from collections.abc import Callable

require_dependency("autograd")
ag_builtins = import_module("autograd.builtins")
ag_extend = import_module("autograd.extend")
ag_tracer = import_module("autograd.tracer")

_HIGHER_ORDER_ERROR = (
    "the HIPS Autograd bridge supports first-order VJPs only; "
    "higher-order differentiation is unsupported"
)


def _contains_box(value: Any) -> bool:
    leaves, _treedef = tree_flatten(value)
    return any(ag_tracer.isbox(leaf) for leaf in leaves)


def _contains_nested_box(value: Any) -> bool:
    leaves, _treedef = tree_flatten(value)
    return any(
        ag_tracer.isbox(leaf) and _contains_box(cast("Any", leaf)._value)  # noqa: SLF001
        for leaf in leaves
    )


def _concrete(value: Any) -> Any:
    leaves, treedef = tree_flatten(value)
    return tree_unflatten(treedef, [ag_tracer.getval(leaf) for leaf in leaves])


def _gradient_like(gradient: Any, primal: Any) -> Any:
    array = np.asarray(gradient)
    primal_array = np.asarray(primal)
    if primal_array.dtype.kind != "c":
        array = np.real(array)
    result = np.asarray(array, dtype=primal_array.dtype)
    if not hasattr(primal, "shape") and result.shape == ():
        return result.item()
    return result


def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a NumPy-backed callable as a first-order HIPS Autograd primitive.

    Every NumPy floating or complex leaf in the positional arguments is
    selected. The bridge translates between Autograd's complex-bilinear
    cotangents and Advect's real-adjoint convention. Higher-order
    differentiation is rejected.
    """

    @functools.wraps(function)
    def wrapped(*args: Any) -> Any:
        if not _contains_box(args):
            numeric_tree(args, boundary="HIPS Autograd bridge input")
            value = function(*args)
            numeric_tree(value, boundary="Advect output")
            return value
        if _contains_nested_box(args):
            raise NotImplementedError(_HIGHER_ORDER_ERROR)

        concrete_inputs = _concrete(args)
        input_leaves, input_treedef = numeric_tree(
            concrete_inputs,
            boundary="HIPS Autograd bridge input",
        )
        output_treedef_holder: list[Any] = []
        pullback_holder: list[Any] = []

        @ag_extend.primitive
        def execute(ordered_values: tuple[Any, ...]) -> tuple[Any, ...]:
            concrete_args = _concrete(ordered_values)
            output_leaves, output_treedef, pullback = validated_vjp(
                function,
                concrete_args,
            )
            output_treedef_holder.append(output_treedef)
            pullback_holder.append(pullback)
            return tuple(output_leaves)

        def make_vjp(
            _answer: Any,
            _ordered_values: tuple[Any, ...],
        ) -> Callable[[Any], Any]:
            pullback = pullback_holder.pop()
            output_treedef = output_treedef_holder[-1]

            def apply(cotangents: Any) -> Any:
                if _contains_box(cotangents):
                    pullback.close()
                    raise NotImplementedError(_HIGHER_ORDER_ERROR)
                cotangent_values = tuple(cotangents)
                cotangent_tree = tree_unflatten(
                    output_treedef,
                    list(cotangent_values),
                )
                gradients = pullback(conjugate_complex_tree(cotangent_tree))
                gradients = conjugate_complex_tree(gradients)
                gradient_leaves, _gradient_treedef = tree_flatten(gradients)
                gradient_tree = tree_unflatten(
                    input_treedef,
                    [
                        _gradient_like(gradient, primal)
                        for gradient, primal in zip(
                            gradient_leaves,
                            input_leaves,
                            strict=True,
                        )
                    ],
                )
                return ag_builtins.tuple(gradient_tree)

            return apply

        ag_extend.defvjp(execute, make_vjp)
        flat_outputs = execute(ag_builtins.tuple(args))
        return tree_unflatten(output_treedef_holder[0], list(flat_outputs))

    return wrapped


__all__ = ["wrap"]
