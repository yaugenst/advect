"""PyTorch reverse-mode bridge for NumPy-backed Advect callables."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from advect.core._pytree import tree_flatten, tree_unflatten
from advect.interop._common import (
    numeric_tree,
    require_dependency,
    validated_vjp,
)

if TYPE_CHECKING:
    from collections.abc import Callable

torch = require_dependency("torch")
once_differentiable = torch.autograd.function.once_differentiable

type _InputSpec = tuple[Any, Any, bool]


def _input_tensor_leaves(args: tuple[Any, ...]) -> tuple[list[Any], Any, Any]:
    leaves, treedef = tree_flatten(args)
    if not leaves:
        message = "PyTorch bridge inputs must contain at least one tensor leaf"
        raise TypeError(message)
    for index, leaf in enumerate(leaves):
        if not torch.is_tensor(leaf):
            message = f"PyTorch bridge input leaf {index} is not a torch.Tensor"
            raise TypeError(message)
        if not (leaf.is_floating_point() or leaf.is_complex()):
            message = (
                f"PyTorch bridge input leaf {index} has dtype {leaf.dtype}; "
                "only floating and complex tensors are supported"
            )
            raise TypeError(message)
    device = leaves[0].device
    if any(leaf.device != device for leaf in leaves[1:]):
        message = "all PyTorch bridge input tensors must be on one device"
        raise ValueError(message)
    return leaves, treedef, device


def _tensor_to_numpy(value: Any) -> np.ndarray:
    try:
        return np.array(value.detach().resolve_conj().cpu().numpy(), copy=True)
    except TypeError as error:
        message = f"PyTorch dtype {value.dtype} cannot cross the NumPy bridge"
        raise TypeError(message) from error


def _output_tensor(value: Any, *, device: Any) -> Any:
    array = np.array(value, copy=True)
    return torch.as_tensor(array, device=device)


def _input_spec(primal: Any) -> _InputSpec:
    return primal.device, primal.dtype, primal.requires_grad


def _gradient_tensor(gradient: Any, spec: _InputSpec) -> Any:
    device, dtype, requires_grad = spec
    if not requires_grad:
        return None
    array = np.asarray(gradient)
    if not dtype.is_complex:
        array = np.real(array)
    return torch.as_tensor(np.array(array, copy=True), device=device, dtype=dtype)


def _direct_call(function: Callable[..., Any], args: tuple[Any, ...], *, device: Any) -> Any:
    leaves, input_treedef = tree_flatten(args)
    concrete_args = tree_unflatten(input_treedef, [_tensor_to_numpy(leaf) for leaf in leaves])
    value = function(*concrete_args)
    output_leaves, output_treedef = numeric_tree(value, boundary="Advect output")
    return tree_unflatten(
        output_treedef,
        [_output_tensor(leaf, device=device) for leaf in output_leaves],
    )


def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a NumPy-backed callable as a first-order PyTorch operation.

    Every tensor leaf with a NumPy floating or complex representation is an
    Advect input. Static configuration should be closed over by ``function``.
    Values execute through NumPy on the host and outputs return to the inputs'
    common device. One PyTorch backward consumes the retained Advect pullback.
    """

    @functools.wraps(function)
    def wrapped(*args: Any) -> Any:
        input_leaves, input_treedef, device = _input_tensor_leaves(args)
        if not torch.is_grad_enabled() or not any(leaf.requires_grad for leaf in input_leaves):
            return _direct_call(function, args, device=device)

        output_treedef_holder: list[Any] = []

        class _AdvectFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *flat_inputs: Any) -> tuple[Any, ...]:
                concrete_args = tree_unflatten(
                    input_treedef,
                    [_tensor_to_numpy(leaf) for leaf in flat_inputs],
                )
                output_leaves, output_treedef, pullback = validated_vjp(
                    function,
                    concrete_args,
                )
                try:
                    outputs = tuple(_output_tensor(leaf, device=device) for leaf in output_leaves)
                except BaseException:
                    pullback.close()
                    raise
                output_treedef_holder.append(output_treedef)
                ctx.set_materialize_grads(False)
                ctx.input_specs = tuple(_input_spec(primal) for primal in flat_inputs)
                ctx.output_treedef = output_treedef
                ctx.pullback = pullback
                return outputs

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, *cotangents: Any) -> tuple[Any, ...]:
                cotangent_tree = tree_unflatten(
                    ctx.output_treedef,
                    [
                        None if cotangent is None else _tensor_to_numpy(cotangent)
                        for cotangent in cotangents
                    ],
                )
                gradients = ctx.pullback(cotangent_tree)
                gradient_leaves, _gradient_treedef = tree_flatten(gradients)
                return tuple(
                    _gradient_tensor(gradient, spec)
                    for gradient, spec in zip(
                        gradient_leaves,
                        ctx.input_specs,
                        strict=True,
                    )
                )

        flat_outputs = _AdvectFunction.apply(*input_leaves)
        output_leaves = flat_outputs if isinstance(flat_outputs, tuple) else (flat_outputs,)
        return tree_unflatten(output_treedef_holder[0], list(output_leaves))

    return wrapped


__all__ = ["wrap"]
