"""Manual dynamic rematerialization."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from advect.core._array_api.providers import _get_array_namespace
from advect.core._context import (
    _get_active_trace_kind,
    _rematerialization_region,
    is_tracing,
)
from advect.core._errors import TracingError
from advect.core._primitive import primitive
from advect.core._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._primitive import Primitive
    from advect.core._pytree import TreeDef


@dataclass(frozen=True, slots=True)
class _CheckpointConfig:
    function: Callable[..., Any]
    treedef: TreeDef


def _call_from_tree(function: Callable[..., Any], treedef: TreeDef, leaves: tuple[Any, ...]) -> Any:
    args, kwargs = tree_unflatten(treedef, list(leaves))
    with _rematerialization_region():
        return function(*args, **kwargs)


def _zero_tangent_like(value: Any) -> Any:
    namespace = _get_array_namespace(value)
    zeros_like = getattr(namespace, "zeros_like", None) if namespace is not None else None
    if callable(zeros_like):
        return zeros_like(value)
    return value * 0


def _restore_output_cotangent(output: Any, cotangent: Any) -> Any:
    """Rebuild a primitive's flat multi-output cotangent as its output pytree."""
    _output_leaves, output_treedef = tree_flatten(output)
    _cotangent_leaves, cotangent_treedef = tree_flatten(cotangent)
    if cotangent_treedef == output_treedef or output_treedef.node_type is None:
        return cotangent
    if isinstance(cotangent, tuple) and len(cotangent) == output_treedef.num_leaves:
        return tree_unflatten(output_treedef, list(cotangent))
    return cotangent


def _build_checkpoint_primitive() -> Primitive[..., Any]:
    @primitive(
        name="advect_internal.checkpoint",
        static_argnames=("config",),
    )
    def implementation(
        payload: object,
        config: _CheckpointConfig,
    ) -> Any:
        leaves, actual_treedef = tree_flatten(payload)
        if actual_treedef != config.treedef:
            msg = "Checkpointed call structure changed while executing its concrete region"
            raise TypeError(msg)
        return _call_from_tree(config.function, config.treedef, tuple(leaves))

    @implementation.def_abstract
    def abstract(
        payload: object,
        config: _CheckpointConfig,
    ) -> object:
        del payload, config
        msg = (
            "ad.checkpoint() currently supports concrete dynamic autodiff only. "
            "Move the checkpoint outside stage(), or stage the uncheckpointed function."
        )
        raise TracingError(msg)

    @implementation.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        config: _CheckpointConfig,
    ) -> Any:
        del output
        from advect.autodiff.api.forward import jvp  # noqa: PLC0415

        def flat_function(*leaves: Any) -> Any:
            return _call_from_tree(
                config.function,
                config.treedef,
                cast("tuple[Any, ...]", leaves),
            )

        active_tangents = tuple(
            _zero_tangent_like(primal) if tangent is None else tangent
            for primal, tangent in zip(primals, tangents, strict=True)
        )
        _value, tangent = jvp(
            flat_function,
            argnums=tuple(range(len(primals))),
        )(
            *primals,
            tangents=active_tangents,
        )
        return tangent

    @implementation.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        config: _CheckpointConfig,
    ) -> tuple[Any, ...]:
        del output
        from advect.autodiff.api.reverse import vjp  # noqa: PLC0415

        def flat_function(*leaves: Any) -> Any:
            return _call_from_tree(
                config.function,
                config.treedef,
                cast("tuple[Any, ...]", leaves),
            )

        value, pullback = vjp(
            flat_function,
            argnums=tuple(range(len(primals))),
        )(*primals)
        try:
            result = pullback(_restore_output_cotangent(value, cotangent))
        finally:
            close = getattr(pullback, "close", None)
            if callable(close):
                close()
        if not isinstance(result, tuple):
            return (result,)
        return cast("tuple[Any, ...]", result)

    return implementation


_checkpoint_operation = _build_checkpoint_primitive()


def checkpoint(function: Callable[..., Any]) -> Callable[..., Any]:
    """Return a dynamic rematerialization wrapper for ``function``.

    An ordinary call invokes ``function`` directly. During concrete autodiff,
    Advect records the whole call as one operation on the outer tape and
    recomputes its body when applying a JVP or transpose instead of retaining
    the body's interior trace.

    Parameters
    ----------
    function
        Pure callable to rematerialize. Its positional arguments, keyword
        arguments, and result may be pytrees. Replaying the same explicit
        inputs must produce the same result; observed mutable state and side
        effects are therefore outside the contract.

    Returns
    -------
    Callable
        A wrapper with the apparent signature and metadata of ``function``.
        Calling it as ``wrapped(*args, **kwargs)`` returns the same output
        pytree as ``function(*args, **kwargs)``.

    Raises
    ------
    TypeError
        If ``function`` is not callable, or if a traced invocation changes its
        input pytree structure while the rematerialized region is executing.
    TracingError
        If abstract staging reaches the wrapper, or if recomputation reaches a
        residual-bearing primitive whose opaque residual cannot cross the
        checkpoint boundary.

    Notes
    -----
    Checkpointing is a concrete dynamic transform; it does not create a
    durable staged region, and `stage` rejects a checkpointed call. Nested
    dynamic derivatives are supported when the recomputed callable and all
    derivative rules on its path remain traceable at the nested level.

    The returned wrapper retains ``function`` and its closure for the
    wrapper's lifetime. Each JVP or transpose application owns and releases
    its temporary inner trace before returning; checkpointing exposes no
    additional resource handle to close.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> @ad.checkpoint
    ... def square(value):
    ...     return value**2
    >>> ad.grad(lambda value: np.sum(square(value)))(np.array([1.0, 2.0, 3.0])).tolist()
    [2.0, 4.0, 6.0]
    """
    if not callable(function):
        msg = "checkpoint function must be callable"
        raise TypeError(msg)

    @functools.wraps(function)
    def checkpointed(*args: Any, **kwargs: Any) -> Any:
        if not is_tracing():
            return function(*args, **kwargs)
        if _get_active_trace_kind() == "stage_abstract":
            msg = (
                "ad.checkpoint() currently supports concrete dynamic autodiff only. "
                "Move the checkpoint outside stage(), or stage the uncheckpointed function."
            )
            raise TracingError(msg)
        payload = (args, kwargs)
        leaves, treedef = tree_flatten(payload)
        if not leaves:
            return function(*args, **kwargs)
        return _checkpoint_operation._call_dynamic_only(  # noqa: SLF001
            payload=payload,
            config=_CheckpointConfig(function, treedef),
        )

    return checkpointed


__all__ = ["checkpoint"]
