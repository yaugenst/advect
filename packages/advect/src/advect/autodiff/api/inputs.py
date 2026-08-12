"""Input selection and pytree tracing helpers for the unified autodiff API."""

from __future__ import annotations

import functools
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from advect.autodiff.api._input_trace import (
    _array_namespace_for_input,
    _LeafTraceSpec,
    _trace_leaf_as_input,
    _TracedInputSpec,
)
from advect.autodiff.api._scalar_boundary import (
    _is_real_python_scalar,
    _lift_scalar_to_array,
)
from advect.autodiff.api.trace import _wrap_input
from advect.core._context import _get_active_trace_level, is_debug
from advect.core._pytree import (
    format_path,
    tree_flatten_with_paths,
    tree_unflatten,
)
from advect.core._stage import StagedProgram

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._pytree import TreePath


def _format_leaf_name(prefix: str | None, path: TreePath) -> str | None:
    if prefix is None:
        return None
    return f"{prefix}{format_path(path)}"


@dataclass(frozen=True, slots=True)
class _SignatureMetadata:
    signature: inspect.Signature
    fixed_positional_names: tuple[str, ...]
    positional_index_by_name: dict[str, int]
    varargs_name: str | None


_SIGNATURE_METADATA_CACHE_SIZE = 512


def _build_signature_metadata(sig: inspect.Signature) -> _SignatureMetadata:
    fixed_positional_names: list[str] = []
    positional_index_by_name: dict[str, int] = {}
    varargs_name: str | None = None
    index = 0

    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            fixed_positional_names.append(param.name)
            positional_index_by_name[param.name] = index
            index += 1
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs_name = param.name
            continue

    return _SignatureMetadata(
        signature=sig,
        fixed_positional_names=tuple(fixed_positional_names),
        positional_index_by_name=positional_index_by_name,
        varargs_name=varargs_name,
    )


def _get_signature_metadata_uncached(f: Callable[..., object]) -> _SignatureMetadata:
    try:
        return _build_signature_metadata(inspect.signature(f))
    except (ValueError, TypeError) as e:
        msg = f"Cannot inspect signature of {f}: {e}"
        raise ValueError(msg) from e


@functools.lru_cache(maxsize=_SIGNATURE_METADATA_CACHE_SIZE)
def _get_signature_metadata_cached(f: Callable[..., object]) -> _SignatureMetadata:
    return _get_signature_metadata_uncached(f)


def _get_signature_metadata(f: Callable[..., object]) -> _SignatureMetadata:
    try:
        return _get_signature_metadata_cached(f)
    except TypeError:
        # Unhashable callable objects cannot be cached by lru_cache keys.
        return _get_signature_metadata_uncached(f)


def _get_signature(f: Callable[..., object]) -> inspect.Signature:
    return _get_signature_metadata(f).signature


def _validate_argnames(sig: inspect.Signature, argnames: tuple[str, ...]) -> None:
    if len(set(argnames)) != len(argnames):
        msg = f"argnames contains duplicates: {argnames}"
        raise ValueError(msg)
    param_names = list(sig.parameters.keys())
    for name in argnames:
        if name not in sig.parameters:
            msg = f"Argument '{name}' not found in function signature. Available: {param_names}"
            raise ValueError(msg)
        kind = sig.parameters[name].kind
        if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            msg = f"argnames cannot select variadic parameter '{name}'"
            raise ValueError(msg)


def _trace_value_as_inputs(
    graph: DynamicTape,
    value: object,
    *,
    prefix: str | None,
    xp: object | None,
) -> tuple[object, _TracedInputSpec]:
    leaf_trace = _trace_leaf_as_input(graph, value, prefix=prefix, xp=xp)
    if leaf_trace is not None:
        return leaf_trace

    paths, leaves, treedef = tree_flatten_with_paths(value)
    traced_leaves: list[Any] = []
    leaf_specs: list[_LeafTraceSpec] = []
    untraceable: list[tuple[TreePath, Any]] = []

    for path, leaf in zip(paths, leaves, strict=True):
        leaf_name = _format_leaf_name(prefix, path)
        if isinstance(leaf, complex):
            label = leaf_name or format_path(path)
            msg = (
                f"Selected input leaf {label!r} is a Python complex scalar, which Advect "
                "does not trace. Wrap it in a backend 0-D array (for example, "
                "numpy.asarray(z)) before differentiation."
            )
            raise TypeError(msg)

        is_existing_traced = callable(getattr(leaf, "_advect_snapshot", None))
        should_trace = is_existing_traced or _is_real_python_scalar(leaf)
        if not should_trace and _array_namespace_for_input(leaf) is not None:
            should_trace = True

        if not should_trace:
            traced_leaves.append(leaf)
            if not isinstance(leaf, (int, bool)):
                untraceable.append((path, leaf))
            leaf_specs.append(
                _LeafTraceSpec(node_id=None, primal=None, restore_python_scalar=False)
            )
            continue

        restore_python_scalar = False
        primal = leaf
        if (not is_existing_traced) and _is_real_python_scalar(leaf):
            primal = _lift_scalar_to_array(leaf, namespace=xp)
            restore_python_scalar = True

        traced, node_id = _wrap_input(
            primal,
            graph,
            name=leaf_name,
            weak=restore_python_scalar or bool(getattr(leaf, "_advect_weak", False)),
        )
        traced_leaves.append(traced)
        leaf_specs.append(
            _LeafTraceSpec(
                node_id=node_id,
                primal=primal,
                restore_python_scalar=restore_python_scalar,
            )
        )

    traced_tree = tree_unflatten(treedef, traced_leaves)
    if is_debug() and untraceable:
        leaf_labels = ", ".join(
            f"{_format_leaf_name(prefix, path) or format_path(path)} ({type(leaf).__name__})"
            for path, leaf in untraceable
        )
        msg = (
            "Encountered untraceable leaf/leaves in a pytree input selected for "
            f"differentiation: {leaf_labels}. They will be treated as static and will "
            "produce None gradients. Wrap leaves with "
            "advect.pytree.static(...) to silence this."
        )
        warnings.warn(msg, stacklevel=2)
    return traced_tree, _TracedInputSpec(treedef=treedef, leaf_specs=tuple(leaf_specs))


def _trace_passive_outer_value(
    graph: DynamicTape,
    value: object,
    *,
    prefix: str,
) -> object:
    """Lift outer tracers into this tape without differentiating their positions."""
    if callable(getattr(value, "_advect_snapshot", None)):
        traced, _node_id = _wrap_input(value, graph, name=prefix, active=False)
        return traced

    paths, leaves, treedef = tree_flatten_with_paths(value)
    traced_leaves: list[Any] | None = None
    for index, (path, leaf) in enumerate(zip(paths, leaves, strict=True)):
        if not callable(getattr(leaf, "_advect_snapshot", None)):
            continue
        if traced_leaves is None:
            traced_leaves = list(leaves)
        leaf_name = _format_leaf_name(prefix, path)
        traced, _node_id = _wrap_input(leaf, graph, name=leaf_name, active=False)
        traced_leaves[index] = traced

    return value if traced_leaves is None else tree_unflatten(treedef, traced_leaves)


def _selected_positional_indices(
    f: Callable[..., Any],
    *,
    normalized_argnums: list[int],
    argnames: tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> set[int]:
    selected = set(normalized_argnums)
    if argnames is None or isinstance(f, StagedProgram):
        return selected

    metadata = _get_signature_metadata(f)
    for name in argnames:
        if name in kwargs:
            continue
        position = metadata.positional_index_by_name.get(name)
        if position is not None:
            selected.add(position)
    return selected


def _trace_passive_outer_arguments(
    f: Callable[..., Any],
    *,
    graph: DynamicTape,
    traced_args: list[Any],
    traced_kwargs: dict[str, Any],
    normalized_argnums: list[int],
    argnames: tuple[str, ...] | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Preserve unselected dependencies when a transform is nested in another trace."""
    trace_level = _get_active_trace_level()
    if trace_level is None or trace_level == 0:
        return traced_args, traced_kwargs

    selected_positions = _selected_positional_indices(
        f,
        normalized_argnums=normalized_argnums,
        argnames=argnames,
        kwargs=traced_kwargs,
    )
    fixed_names, varargs_name = _get_positional_param_names(f)
    for position, value in enumerate(traced_args):
        if position in selected_positions:
            continue
        prefix = _prefix_for_argnum(position, fixed=fixed_names, varargs_name=varargs_name)
        traced_args[position] = _trace_passive_outer_value(
            graph,
            value,
            prefix=prefix,
        )

    selected_names = frozenset(() if argnames is None else argnames)
    passive_kwargs: dict[str, Any] | None = None
    for name, value in traced_kwargs.items():
        if name in selected_names:
            continue
        traced = _trace_passive_outer_value(graph, value, prefix=name)
        if traced is value:
            continue
        if passive_kwargs is None:
            passive_kwargs = dict(traced_kwargs)
        passive_kwargs[name] = traced
    return traced_args, traced_kwargs if passive_kwargs is None else passive_kwargs


def _normalize_argnums_spec(argnums: int | tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
    """Normalize the argnums spec to a tuple and return whether it was singular."""
    if isinstance(argnums, int):
        return (argnums,), True
    return argnums, False


def _normalize_argnums_for_call(argnums: tuple[int, ...], *, nargs: int) -> list[int]:
    """Normalize argnums for a specific call and validate they are in-range and unique."""
    normalized_argnums: list[int] = []
    for argnum in argnums:
        normalized = argnum if argnum >= 0 else nargs + argnum
        if normalized < 0 or normalized >= nargs:
            msg = f"argnums index {argnum} is out of range for {nargs} positional arguments"
            raise IndexError(msg)
        normalized_argnums.append(normalized)

    if len(set(normalized_argnums)) != len(normalized_argnums):
        msg = f"argnums contains duplicates: {argnums}"
        raise ValueError(msg)
    return normalized_argnums


def _get_positional_param_names(f: Callable[..., Any]) -> tuple[tuple[str, ...], str | None]:
    try:
        metadata = _get_signature_metadata(f)
    except ValueError:
        return (), None

    return metadata.fixed_positional_names, metadata.varargs_name


def _prefix_for_argnum(argnum: int, *, fixed: tuple[str, ...], varargs_name: str | None) -> str:
    if argnum < len(fixed):
        return fixed[argnum]
    if varargs_name is not None:
        return f"{varargs_name}[{argnum - len(fixed)}]"
    return f"arg{argnum}"


def _check_argnums_argnames_overlap(
    f: Callable[..., Any],
    *,
    normalized_argnums: list[int],
    argnames: tuple[str, ...] | None,
) -> None:
    if argnames is None or not normalized_argnums:
        return
    if isinstance(f, StagedProgram):
        # A staged call artifact distinguishes positional and keyword leaves
        # structurally. It cannot receive one runtime value through both paths.
        return

    metadata = _get_signature_metadata(f)
    _validate_argnames(metadata.signature, argnames)

    for argnum in normalized_argnums:
        if argnum >= len(metadata.fixed_positional_names):
            continue
        name = metadata.fixed_positional_names[argnum]
        if name in argnames:
            msg = f"Argument '{name}' was selected by both argnums and argnames"
            raise ValueError(msg)


def _trace_positional_selections(
    f: Callable[..., Any],
    *,
    graph: DynamicTape,
    args: tuple[Any, ...],
    normalized_argnums: list[int],
    xp: object | None,
) -> tuple[list[Any], list[_TracedInputSpec]]:
    traced_args = list(args)
    pos_specs: list[_TracedInputSpec] = []

    fixed_names, varargs_name = _get_positional_param_names(f)

    for argnum in normalized_argnums:
        prefix = _prefix_for_argnum(argnum, fixed=fixed_names, varargs_name=varargs_name)
        traced_value, spec = _trace_value_as_inputs(
            graph,
            args[argnum],
            prefix=prefix,
            xp=xp,
        )
        traced_args[argnum] = traced_value
        pos_specs.append(spec)

    return traced_args, pos_specs


def _trace_named_selections(
    f: Callable[..., Any],
    *,
    graph: DynamicTape,
    traced_args: list[Any],
    traced_kwargs: dict[str, Any],
    argnames: tuple[str, ...] | None,
    xp: object | None,
) -> tuple[
    list[Any],
    dict[str, Any],
    dict[str, _TracedInputSpec],
]:
    if argnames is None:
        return traced_args, traced_kwargs, {}

    if isinstance(f, StagedProgram):
        named_specs: dict[str, _TracedInputSpec] = {}
        for name in argnames:
            if name not in traced_kwargs:
                msg = (
                    f"Staged argument '{name}' was not provided as a keyword. "
                    "Select positional staged inputs with argnums."
                )
                raise ValueError(msg)
            traced_value, spec = _trace_value_as_inputs(
                graph,
                traced_kwargs[name],
                prefix=name,
                xp=xp,
            )
            traced_kwargs[name] = traced_value
            named_specs[name] = spec
        return traced_args, traced_kwargs, named_specs

    metadata = _get_signature_metadata(f)
    _validate_argnames(metadata.signature, argnames)

    named_specs: dict[str, _TracedInputSpec] = {}

    for name in argnames:
        if name in traced_kwargs:
            value = traced_kwargs[name]
            traced_value, spec = _trace_value_as_inputs(
                graph,
                value,
                prefix=name,
                xp=xp,
            )
            traced_kwargs[name] = traced_value
        else:
            pos_index = metadata.positional_index_by_name.get(name)
            if pos_index is None or pos_index >= len(traced_args):
                msg = (
                    f"Argument '{name}' was not provided in the call. "
                    "Pass it positionally or as a keyword argument."
                )
                raise ValueError(msg)
            traced_value, spec = _trace_value_as_inputs(
                graph,
                traced_args[pos_index],
                prefix=name,
                xp=xp,
            )
            traced_args[pos_index] = traced_value

        named_specs[name] = spec

    return traced_args, traced_kwargs, named_specs


def _trace_selected_args_and_kwargs(
    f: Callable[..., Any],
    *,
    graph: DynamicTape,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    normalized_argnums: list[int],
    argnames: tuple[str, ...] | None,
    xp: object | None,
) -> tuple[
    list[Any],
    dict[str, Any],
    list[_TracedInputSpec],
    dict[str, _TracedInputSpec],
]:
    """Trace selected positional args and argnames (pytree-aware)."""
    _check_argnums_argnames_overlap(f, normalized_argnums=normalized_argnums, argnames=argnames)

    traced_args, pos_specs = _trace_positional_selections(
        f,
        graph=graph,
        args=args,
        normalized_argnums=normalized_argnums,
        xp=xp,
    )
    traced_kwargs = kwargs if argnames is None else dict(kwargs)

    (
        traced_args,
        traced_kwargs,
        named_specs,
    ) = _trace_named_selections(
        f,
        graph=graph,
        traced_args=traced_args,
        traced_kwargs=traced_kwargs,
        argnames=argnames,
        xp=xp,
    )
    traced_args, traced_kwargs = _trace_passive_outer_arguments(
        f,
        graph=graph,
        traced_args=traced_args,
        traced_kwargs=traced_kwargs,
        normalized_argnums=normalized_argnums,
        argnames=argnames,
    )
    return traced_args, traced_kwargs, pos_specs, named_specs
