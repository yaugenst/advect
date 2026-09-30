# ruff: noqa: ANN401  # Primitive values are intentionally backend-generic.
"""Dynamic call representation and tracing for user-authored primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from advect.core._array_api.providers import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_protocol_helpers import literal_is_weak, weak_scalar_runtime_value
from advect.core._backends import get_hook
from advect.core._context import (
    _is_recorder_in_active_trace_stack,
    _select_deepest_active_recorder,
    _suspend_tracing,
    _trace_frame_for_recorder,
    get_source_location,
)
from advect.core._errors import TracingError
from advect.core._graph_attrs import _PRIMITIVE_CALL_KEY
from advect.core._protocols import ArrayLike, _is_traced, _snapshot_traced
from advect.core._pytree import (
    DictKey,
    SequenceKey,
    _tree_contains_tracer,
    format_path,
    tree_flatten,
    tree_flatten_with_paths,
    tree_unflatten,
)
from advect.core._registry import get_registry
from advect.core._residual import _PrimitiveExecution
from advect.core._stage_serialization import (
    _decode_treedef,
    _decode_value,
    _encode_treedef,
    _encode_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from advect.core._native import DynamicTape
    from advect.core._pytree import TreeDef, TreePath

_CALL_TREE_LEN = 2
_KEYWORD_PATH_LEN = 2
_PRIMITIVE_CALL_FIELDS = {
    "call_treedef",
    "input_leaf_mask",
    "static_leaves",
    "output_treedef",
    "nondiff_input_mask",
}


@dataclass(frozen=True, slots=True)
class _PrimitiveCallMeta:
    """Closed call structure stored on a dynamic primitive node."""

    call_treedef: TreeDef
    input_leaf_mask: tuple[bool, ...]
    static_leaves: tuple[Any, ...]
    output_treedef: TreeDef
    nondiff_input_mask: tuple[bool, ...] = ()

    def nondiff_mask(self, input_count: int) -> tuple[bool, ...]:
        if not self.nondiff_input_mask:
            return (False,) * input_count
        if len(self.nondiff_input_mask) != input_count:
            msg = (
                "Primitive nondifferentiable mask does not match node inputs: "
                f"expected {input_count}, got {len(self.nondiff_input_mask)}"
            )
            raise TypeError(msg)
        return self.nondiff_input_mask


def _encode_bool_mask(value: tuple[bool, ...]) -> list[bool]:
    if any(type(item) is not bool for item in value):
        msg = "Primitive call masks must contain exact booleans"
        raise TypeError(msg)
    return list(value)


def _decode_bool_mask(value: object, *, label: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        msg = f"Encoded primitive {label} must be a list of exact booleans"
        raise TypeError(msg)
    return tuple(value)


def _encode_primitive_call_meta(value: object) -> object:
    """Encode one closed primitive call contract for native graph ownership."""
    if not isinstance(value, _PrimitiveCallMeta):
        msg = "Primitive call metadata has an invalid runtime value"
        raise TypeError(msg)
    if len(value.input_leaf_mask) != value.call_treedef.num_leaves:
        msg = "Primitive input mask does not match its call pytree"
        raise ValueError(msg)
    input_count = sum(value.input_leaf_mask)
    if len(value.static_leaves) != value.call_treedef.num_leaves - input_count:
        msg = "Primitive static leaves do not match its call pytree"
        raise ValueError(msg)
    value.nondiff_mask(input_count)
    return {
        "call_treedef": _encode_treedef(value.call_treedef),
        "input_leaf_mask": _encode_bool_mask(value.input_leaf_mask),
        "static_leaves": [_encode_value(item) for item in value.static_leaves],
        "output_treedef": _encode_treedef(value.output_treedef),
        "nondiff_input_mask": _encode_bool_mask(value.nondiff_input_mask),
    }


def _decode_primitive_call_meta(value: object) -> object:
    """Decode one durable primitive call contract into its runtime form."""
    if not isinstance(value, dict):
        msg = "Encoded primitive call metadata must be a mapping"
        raise TypeError(msg)
    if set(value) != _PRIMITIVE_CALL_FIELDS:
        msg = "Encoded primitive call metadata has invalid fields"
        raise ValueError(msg)
    raw_static_leaves = value["static_leaves"]
    if not isinstance(raw_static_leaves, list):
        msg = "Encoded primitive static leaves must be a list"
        raise TypeError(msg)
    meta = _PrimitiveCallMeta(
        call_treedef=_decode_treedef(value["call_treedef"]),
        input_leaf_mask=_decode_bool_mask(
            value["input_leaf_mask"],
            label="input leaf mask",
        ),
        static_leaves=tuple(_decode_value(item) for item in raw_static_leaves),
        output_treedef=_decode_treedef(value["output_treedef"]),
        nondiff_input_mask=_decode_bool_mask(
            value["nondiff_input_mask"],
            label="nondifferentiable input mask",
        ),
    )
    # Reuse the encoder's complete structural validation without retaining its
    # temporary wire payload.
    _encode_primitive_call_meta(meta)
    return meta


def _split_primitive_attrs(
    attrs: Mapping[str, Any],
) -> tuple[_PrimitiveCallMeta, dict[str, Any]]:
    meta = attrs.get(_PRIMITIVE_CALL_KEY)
    if not isinstance(meta, _PrimitiveCallMeta):
        msg = "Primitive node is missing its internal dynamic-call metadata"
        raise TypeError(msg)
    node_attrs = dict(attrs)
    node_attrs.pop(_PRIMITIVE_CALL_KEY)
    return meta, node_attrs


def _leaf_to_dynamic_operand(
    recorder: DynamicTape,
    leaf: Any,
) -> tuple[int | None, Any] | None:
    if _is_traced(leaf):
        leaf_recorder = getattr(leaf, "recorder", None)
        if leaf_recorder is recorder:
            node_id, value = _snapshot_traced(leaf)
            return node_id, weak_scalar_runtime_value(leaf, value)
        if leaf_recorder is None or not _is_recorder_in_active_trace_stack(leaf_recorder):
            msg = "Cannot mix traced values from unrelated or expired trace contexts"
            raise TracingError(msg)
        _snapshot_traced(leaf)
        # Recorder-local SSA identifiers cannot cross trace levels. Retain an
        # enclosing tracer as an opaque literal so evaluation on the inner
        # tape still records its dependence in the enclosing recorder.
        return None, leaf
    if isinstance(leaf, ArrayLike):
        return None, leaf
    if isinstance(leaf, (bool, int, float, complex)):
        return None, leaf
    return None


def _normalize_output_leaf(value: Any, *, namespace: Any | None) -> Any:
    if _is_traced(value):
        # An inner transform may execute the atomic primal with outer tracers.
        return value
    if isinstance(value, ArrayLike):
        return value
    if namespace is not None:
        asarray = getattr(namespace, "asarray", None)
        if callable(asarray) and type(value) in (bool, int, float, complex):
            return asarray(value)
    if type(value) in (bool, int, float, complex):
        return value
    msg = f"Primitive output leaves must be arrays/scalars, got {type(value).__name__}"
    raise TypeError(msg)


def _normalize_output_pytree(
    value: Any,
    *,
    namespace: Any | None,
) -> tuple[list[Any], TreeDef]:
    leaves, treedef = tree_flatten(value)
    if treedef.num_leaves < 1:
        msg = "Primitives must return at least one scalar/array leaf"
        raise TypeError(msg)

    normalized: list[Any] = []
    invalid: list[int] = []
    for index, leaf in enumerate(leaves):
        try:
            normalized.append(_normalize_output_leaf(leaf, namespace=namespace))
        except TypeError:
            invalid.append(index)
    if invalid:
        # Paths are only needed to report the failure.
        paths, _leaves, _treedef = tree_flatten_with_paths(value)
        labels = ", ".join(
            f"{format_path(paths[index])} ({type(leaves[index]).__name__})" for index in invalid
        )
        msg = f"Primitive returned invalid output leaf/leaves: {labels}"
        raise TypeError(msg)
    return normalized, treedef


def _validate_output_treedef(
    meta: _PrimitiveCallMeta,
    treedef: TreeDef,
    *,
    op: str,
) -> None:
    _validate_output_treedef_against(meta.output_treedef, treedef, op=op)


def _validate_output_treedef_against(
    expected: TreeDef,
    actual: TreeDef,
    *,
    op: str,
) -> None:
    """Validate a concrete primitive output against its traced public structure."""
    if actual == expected:
        return
    msg = (
        f"Primitive '{op.removeprefix('custom.')}' returned an output pytree with a "
        "different structure than it returned while tracing"
    )
    raise ValueError(msg)


def _reconstruct_primitive_output(
    meta: _PrimitiveCallMeta,
    value: object,
    *,
    label: str,
) -> object:
    """Reconstruct one public output pytree from its physical tape value."""
    if meta.output_treedef.node_type is None:
        return value
    leaf_count = meta.output_treedef.num_leaves
    if leaf_count == 1:
        leaves = [value]
    elif isinstance(value, tuple) and len(value) == leaf_count:
        leaves = list(value)
    else:
        msg = (
            f"Primitive {label} storage does not match its output pytree: "
            f"expected {leaf_count} leaves"
        )
        raise TypeError(msg)
    return tree_unflatten(meta.output_treedef, leaves)


def _flatten_primitive_output(
    meta: _PrimitiveCallMeta,
    value: object,
    *,
    label: str,
) -> object:
    """Flatten one authored output pytree into the tape's physical convention."""
    leaves, treedef = tree_flatten(value)
    if treedef != meta.output_treedef:
        msg = f"Primitive {label} must match the primitive output pytree"
        raise ValueError(msg)
    if len(leaves) == 1:
        return leaves[0]
    return tuple(leaves)


def _infer_namespace(values: tuple[Any, ...] | list[Any]) -> Any | None:
    for value in values:
        namespace = _get_array_namespace(value)
        if namespace is not None:
            return namespace
    return None


def _unflatten_call_tree(
    treedef: TreeDef,
    leaves: list[Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    call_tree = tree_unflatten(treedef, leaves)
    if not isinstance(call_tree, tuple) or len(call_tree) != _CALL_TREE_LEN:
        msg = "Internal error: primitive call metadata did not reconstruct (args, kwargs)"
        raise TypeError(msg)
    args, kwargs = call_tree
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        msg = "Internal error: primitive call metadata has an invalid root structure"
        raise TypeError(msg)
    return args, kwargs


def _reconstruct_primitive_call(
    meta: _PrimitiveCallMeta,
    inputs: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    input_iter = iter(inputs)
    static_iter = iter(meta.static_leaves)
    leaves = [
        next(input_iter) if is_input else next(static_iter) for is_input in meta.input_leaf_mask
    ]
    return _unflatten_call_tree(meta.call_treedef, leaves)


def _flatten_input_gradients(
    result: object,
    *,
    expected_input_count: int,
) -> tuple[object | None, ...]:
    """Validate the public flat transpose-result contract."""
    if not isinstance(result, tuple):
        msg = "Primitive transpose rule must return a tuple of gradients"
        raise TypeError(msg)
    leaves, treedef = tree_flatten(result)
    _expected_leaves, expected_treedef = tree_flatten(tuple(range(expected_input_count)))
    if treedef != expected_treedef:
        msg = (
            "Primitive transpose result must be a flat tuple with one contribution "
            "per dynamic input leaf"
        )
        raise ValueError(msg)
    return tuple(leaves)


def _keyword_parameter(path: TreePath) -> str | None:
    if len(path) < _KEYWORD_PATH_LEN:
        return None
    root, parameter = path[0], path[1]
    if not isinstance(root, SequenceKey) or root.index != 1:
        return None
    if not isinstance(parameter, DictKey) or not isinstance(parameter.key, str):
        return None
    return parameter.key


@dataclass(frozen=True, slots=True)
class _TracedCall:
    """Dynamic operands of one primitive call on its recording tape."""

    node_ids: tuple[int, ...]
    parent_positions: tuple[int, ...]
    literals: tuple[Any, ...]
    call_treedef: TreeDef
    nondiff_mask: tuple[bool, ...]
    kwargs: dict[str, Any]
    namespace: Any | None
    outer_recorders: tuple[object, ...]


def _trace_call_arguments(
    recorder: DynamicTape,
    *,
    op_name: str,
    kwargs: dict[str, Any],
    nondiff_argnames: frozenset[str],
) -> _TracedCall:
    """Substitute every dynamic-argument leaf with its operand on ``recorder``.

    The durable call tree keeps its ``(args, kwargs)`` root with empty
    positional arguments. Every leaf belongs to a dynamic argument, so a leaf
    that is not an array, scalar, or tracer is rejected.
    """
    leaves, call_treedef = tree_flatten(((), kwargs))
    # Name leaves by the flattened keys: a re-registered dict node may reorder them.
    kwargs_def = call_treedef.children[1]
    parameters = (
        name
        for name, argument_def in zip(kwargs_def.aux_data, kwargs_def.children, strict=True)
        for _ in range(argument_def.num_leaves)
    )
    node_ids: list[int] = []
    parent_positions: list[int] = []
    literals: list[Any] = []
    nondiff_mask: list[bool] = []
    values: list[Any] = []
    outer_recorders: list[object] = []

    for parameter, leaf in zip(parameters, leaves, strict=True):
        dynamic_operand = _leaf_to_dynamic_operand(recorder, leaf)
        if dynamic_operand is None:
            msg = (
                f"Primitive '{op_name.removeprefix('custom.')}' argument '{parameter}' "
                "is not traceable; declare it in static_argnames or pass an array/scalar"
            )
            raise TypeError(msg)
        node_id, value = dynamic_operand
        if node_id is None:
            literals.append(value)
        else:
            node_ids.append(node_id)
            parent_positions.append(len(values))
        nondiff_mask.append(parameter in nondiff_argnames)
        values.append(value)
        # An enclosing tracer, passed directly or as an inner tracer's payload,
        # routes execution through the enclosing recorder.
        if (
            _is_traced(value)
            and (value_recorder := getattr(value, "recorder", None)) is not None
            and value_recorder is not recorder
        ):
            outer_recorders.append(value_recorder)

    _call_args, call_kwargs = _unflatten_call_tree(call_treedef, values)
    return _TracedCall(
        node_ids=tuple(node_ids),
        parent_positions=tuple(parent_positions),
        literals=tuple(literals),
        call_treedef=call_treedef,
        nondiff_mask=tuple(nondiff_mask),
        kwargs=call_kwargs,
        namespace=_infer_namespace(leaves),
        outer_recorders=tuple(outer_recorders),
    )


def _output_shape_and_dtype(value: Any) -> tuple[tuple[int, ...], Any]:
    if isinstance(value, ArrayLike):
        return value.shape, value.dtype
    return (
        (),
        {
            bool: "bool",
            int: "int64",
            float: "float64",
            complex: "complex128",
        }.get(type(value), "float64"),
    )


def _record_primitive_output_count(op_name: str, count: int) -> None:
    registry = get_registry()
    op_def = registry.get(op_name)
    if not op_def.output_arity_known:
        registry.update(op_name, num_outputs=count, output_arity_known=True)
        return
    if op_def.num_outputs == count:
        return
    msg = (
        f"Primitive '{op_name.removeprefix('custom.')}' changed its output count "
        f"from {op_def.num_outputs} to {count}"
    )
    raise ValueError(msg)


def _attach_residual(
    recorder: DynamicTape,
    node_id: int,
    execution: _PrimitiveExecution,
) -> None:
    residual = execution.take_residual()
    if residual is None:
        return
    try:
        recorder.record_residual(node_id, residual)
    except Exception:
        residual.close()
        raise


def _weak_output_leaves(recorder: DynamicTape, call: _TracedCall, output: Any) -> list[bool]:
    """Return which output leaves are weak scalars (NEP 50).

    As a Python operator does, an implementation that returns a Python scalar
    from weak operands returns a weak scalar; in a nested trace the enclosing
    trace's value already carries that category.
    """
    weak_operands = all(recorder.is_weak(node_id) for node_id in call.node_ids) and all(
        literal_is_weak(literal) for literal in call.literals
    )
    return [weak_operands and literal_is_weak(leaf) for leaf in tree_flatten(output)[0]]


def trace_primitive_call(  # noqa: PLR0913 - one call carries the complete primitive contract
    function: Callable[..., Any],
    *,
    abstract_function: Callable[..., Any] | None,
    op_name: str,
    schema_version: int,
    recorder: DynamicTape,
    kwargs: dict[str, Any],
    node_attrs: Mapping[str, Any],
    nondiff_argnames: frozenset[str],
    has_residual: bool,
    track_output_arity: bool = True,
) -> Any:
    """Execute one concrete primitive call and append its atomic tape node.

    ``kwargs`` holds exactly the call's dynamic arguments by name.
    """
    if _PRIMITIVE_CALL_KEY in kwargs:
        msg = f"Keyword argument '{_PRIMITIVE_CALL_KEY}' is reserved for Advect internals"
        raise TypeError(msg)
    call = _trace_call_arguments(
        recorder,
        op_name=op_name,
        kwargs=kwargs,
        nondiff_argnames=nondiff_argnames,
    )
    call_kwargs = call.kwargs
    namespace = call.namespace
    outer_recorder = (
        _select_deepest_active_recorder(call.outer_recorders) if call.outer_recorders else None
    )
    if has_residual and outer_recorder is not None:
        msg = (
            f"Primitive '{op_name.removeprefix('custom.')}' uses an opaque residual "
            "and supports first-order differentiation only; it cannot be embedded "
            "in a staged or higher-order derivative"
        )
        raise TracingError(msg)

    direct_execution = outer_recorder is None
    if direct_execution:
        with _suspend_tracing():
            execution = function(**call_kwargs)
    else:
        outer_frame = _trace_frame_for_recorder(outer_recorder)
        if outer_frame is None:
            msg = "Primitive operands belong to an inactive enclosing trace"
            raise TracingError(msg)
        if outer_frame.trace_kind == "stage_abstract":
            if abstract_function is None:
                msg = (
                    f"Primitive '{op_name.removeprefix('custom.')}' cannot be "
                    "preserved in an enclosing staged trace without abstract evaluation"
                )
                raise TracingError(msg)
            nested_output = abstract_function(**call_kwargs)
        else:
            nested_output = trace_primitive_call(
                function,
                abstract_function=abstract_function,
                op_name=op_name,
                schema_version=schema_version,
                recorder=cast("DynamicTape", outer_recorder),
                kwargs=call_kwargs,
                node_attrs=node_attrs,
                nondiff_argnames=nondiff_argnames,
                has_residual=has_residual,
                track_output_arity=track_output_arity,
            )
        execution = _PrimitiveExecution(nested_output, None)
    if not isinstance(execution, _PrimitiveExecution):
        msg = "Internal primitive forward did not return an execution record"
        raise TypeError(msg)
    try:
        if direct_execution and _tree_contains_tracer(execution.output):
            msg = (
                f"Primitive '{op_name.removeprefix('custom.')}' returned a captured "
                "tracer from its implementation. Pass every dynamic "
                "dependency as an explicit primitive argument."
            )
            raise TracingError(msg)
        weak_leaves = _weak_output_leaves(recorder, call, execution.output)
        result_leaves, output_treedef = _normalize_output_pytree(
            execution.output,
            namespace=namespace,
        )
        if direct_execution and not call.node_ids and all(weak_leaves):
            # Weak scalars computed from weak constants have no array provider
            # to trace them; they are constants of the trace, as eagerly.
            return execution.output
        meta = _PrimitiveCallMeta(
            call_treedef=call.call_treedef,
            input_leaf_mask=(True,) * len(call.nondiff_mask),
            static_leaves=(),
            output_treedef=output_treedef,
            nondiff_input_mask=call.nondiff_mask,
        )
        attrs = dict(node_attrs)
        attrs[_PRIMITIVE_CALL_KEY] = meta
        source_location = get_source_location()
        if track_output_arity:
            _record_primitive_output_count(op_name, len(result_leaves))

        single_output = len(result_leaves) == 1
        shapes_dtypes = [_output_shape_and_dtype(leaf) for leaf in result_leaves]
        node_id = recorder.record_operation(
            op_name,
            call.node_ids,
            result_leaves[0] if single_output else tuple(result_leaves),
            attrs,
            *shapes_dtypes[0],
            input_positions=call.parent_positions,
            literals=call.literals,
            weak=single_output and weak_leaves[0],
            schema_version=schema_version,
            source_location=source_location,
        )
        if single_output:
            traced = _wrap_traced_output(
                result_leaves[0],
                node_id=node_id,
                recorder=recorder,
                namespace=namespace,
            )
            _attach_residual(recorder, node_id, execution)
            return tree_unflatten(output_treedef, [traced])

        traced_leaves: list[Any] = []
        for index, (leaf, (shape, dtype)) in enumerate(
            zip(result_leaves, shapes_dtypes, strict=True)
        ):
            output_id = recorder.record_operation(
                "advect.getoutput",
                (node_id,),
                leaf,
                {"index": index, "num_outputs": len(result_leaves)},
                shape,
                dtype,
                weak=weak_leaves[index],
                source_location=source_location,
            )
            traced_leaves.append(
                _wrap_traced_output(
                    leaf,
                    node_id=output_id,
                    recorder=recorder,
                    namespace=namespace,
                )
            )
        _attach_residual(recorder, node_id, execution)
        return tree_unflatten(output_treedef, traced_leaves)
    finally:
        execution.close()


def _wrap_traced_output(
    value: Any,
    *,
    node_id: int,
    recorder: DynamicTape,
    namespace: Any | None,
) -> Any:
    if isinstance(value, ArrayLike):
        resolved_namespace = namespace or _get_array_namespace(value)
        backend = (
            _get_backend_key_from_namespace(resolved_namespace)
            if resolved_namespace is not None
            else None
        )
        wrap_traced = get_hook(f"{backend}.wrap_traced") if backend is not None else None
        if wrap_traced is None and resolved_namespace is not None:
            wrap_traced = get_hook("advect.array_api.wrap_traced")
        if wrap_traced is None:
            msg = "The primitive result's array provider does not support Advect tracing"
            raise RuntimeError(msg)
        return wrap_traced(value, node_id=node_id, recorder=recorder)
    msg = (
        "A traced primitive returned a scalar without an array provider. "
        "Primitive outputs must remain provider-backed during differentiation."
    )
    raise TypeError(msg)
