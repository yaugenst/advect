# ruff: noqa: ANN401  # HIPS Autograd exposes tracer boxes and primitive callbacks dynamically.
"""HIPS Autograd reverse-mode bridge for Advect callables."""

from __future__ import annotations

import functools
import itertools
import weakref
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import advect as ad
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.interop._common import (
    conjugate_complex_tree,
    invoke_with_keywords,
    numeric_tree,
    require_dependency,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._pytree import TreeDef

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


def _gradient_like(gradient: Any, primal: Any) -> Any:
    array = np.asarray(gradient)
    primal_array = np.asarray(primal)
    if primal_array.dtype.kind != "c":
        array = np.real(array)
    result = np.asarray(array, dtype=primal_array.dtype)
    if not hasattr(primal, "shape") and result.shape == ():
        return result.item()
    return result


@dataclass(slots=True)
class _Invocation:
    """Per-call state passed to the shared primitive as a non-boxed argument.

    The arguments flatten to host leaves, where an Autograd container box stays
    one leaf; each host leaf's concrete value flattens to numeric input leaves.
    """

    function: Callable[..., object]
    argument_treedef: TreeDef
    host_treedefs: list[TreeDef]
    input_leaves: list[object]
    output_treedef: TreeDef | None = None
    linear: ad.LinearMap | None = None


@ag_extend.primitive
def _execute(host_leaves: tuple[Any, ...], invocation: _Invocation) -> tuple[Any, ...]:
    concrete_values = tree_unflatten(invocation.argument_treedef, list(host_leaves))
    value, linear = ad.linearize(
        invocation.function,
        *concrete_values,
        argnums=tuple(range(len(concrete_values))),
    )
    try:
        output_leaves, output_treedef = numeric_tree(value, boundary="Advect output")
    except BaseException:
        linear.close()
        raise
    invocation.output_treedef = output_treedef
    invocation.linear = linear
    return tuple(output_leaves)


def _make_vjp(
    _answer: Any,
    _host_leaves: tuple[Any, ...],
    invocation: _Invocation,
) -> Callable[[Any], Any]:
    linear = invocation.linear
    if linear is None:
        # A second trace recorded this call, so it differentiates a derivative.
        raise NotImplementedError(_HIGHER_ORDER_ERROR)
    invocation.linear = None
    output_treedef = cast("TreeDef", invocation.output_treedef)
    input_leaves = invocation.input_leaves
    host_treedefs = invocation.host_treedefs

    def apply(cotangents: Any) -> Any:
        if _contains_box(cotangents):
            linear.close()
            raise NotImplementedError(_HIGHER_ORDER_ERROR)
        try:
            cotangent_tree = tree_unflatten(output_treedef, list(cotangents))
            normalized_cotangent = conjugate_complex_tree(cotangent_tree)
            gradients = conjugate_complex_tree(linear.pullback(normalized_cotangent))
            gradient_leaves, _gradient_treedef = tree_flatten(gradients)
            projected = iter(
                [
                    _gradient_like(gradient, primal)
                    for gradient, primal in zip(gradient_leaves, input_leaves, strict=True)
                ]
            )
            # One gradient per host leaf, structured like that leaf's value.
            return ag_builtins.tuple(
                tree_unflatten(treedef, list(itertools.islice(projected, treedef.num_leaves)))
                for treedef in host_treedefs
            )
        except BaseException:
            linear.close()
            raise

    weakref.finalize(apply, linear.close)
    return apply


# One registration for the process: Autograd's VJP table is global and never
# pruned, so per-call state travels as the non-differentiated argument.
ag_extend.defvjp(_execute, _make_vjp)


def wrap(function: Callable[..., object]) -> Callable[..., object]:
    """Wrap a NumPy-backed callable as a first-order HIPS Autograd primitive.

    Every NumPy floating or complex leaf in positional or keyword arguments is
    selected. The bridge translates between Autograd's complex-bilinear
    cotangents and Advect's real-adjoint convention. The exact forward
    linearization remains reusable for first-order host transforms; higher-order
    differentiation is rejected.
    """

    @functools.wraps(function)
    def wrapped(*args: object, **kwargs: object) -> object:
        values = (*args, *kwargs.values())
        call = functools.partial(
            invoke_with_keywords,
            function,
            positional_count=len(args),
            keyword_names=tuple(kwargs),
        )
        host_leaves, argument_treedef = tree_flatten(values)
        if not any(ag_tracer.isbox(leaf) for leaf in host_leaves):
            numeric_tree(values, boundary="HIPS Autograd bridge input")
            value = call(*values)
            numeric_tree(value, boundary="Advect output")
            return value
        if any(
            ag_tracer.isbox(leaf) and _contains_box(leaf._value)  # noqa: SLF001
            for leaf in host_leaves
        ):
            raise NotImplementedError(_HIGHER_ORDER_ERROR)

        concrete_leaves = [ag_tracer.getval(leaf) for leaf in host_leaves]
        input_leaves, _input_treedef = numeric_tree(
            tree_unflatten(argument_treedef, concrete_leaves),
            boundary="HIPS Autograd bridge input",
        )
        invocation = _Invocation(
            call,
            argument_treedef,
            [tree_flatten(leaf)[1] for leaf in concrete_leaves],
            input_leaves,
        )
        # Autograd boxes a container only when its direct items are boxes, so a
        # dict or list built inside the traced function must pass as host leaves.
        flat_outputs = _execute(ag_builtins.tuple(host_leaves), invocation)
        return tree_unflatten(cast("TreeDef", invocation.output_treedef), list(flat_outputs))

    return wrapped


__all__ = ["wrap"]
