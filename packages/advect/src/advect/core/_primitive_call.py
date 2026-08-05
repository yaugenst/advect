# ruff: noqa: ANN401  # Primitive values are intentionally backend-generic.
"""Dynamic call representation and tracing for user-authored primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from advect.core._array_namespace import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_protocol_helpers import (
    literals_are_weak,
    weak_scalar_runtime_value,
)
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
from advect.core._protocols import ArrayLike, _snapshot_traced
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


def _is_traced_leaf(value: Any) -> bool:
    return callable(getattr(value, "_advect_snapshot", None))


def _leaf_to_dynamic_operand(
    recorder: DynamicTape,
    leaf: Any,
) -> tuple[int | None, Any] | None:
    if _is_traced_leaf(leaf):
        leaf_recorder = getattr(leaf, "recorder", None)
        if leaf_recorder is recorder:
            node_id, value = _snapshot_traced(leaf)
            return int(node_id), weak_scalar_runtime_value(leaf, value)
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
    if _is_traced_leaf(value):
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
    paths, leaves, treedef = tree_flatten_with_paths(value)
    if treedef.num_leaves < 1:
        msg = "Primitives must return at least one scalar/array leaf"
        raise TypeError(msg)

    normalized: list[Any] = []
    invalid: list[tuple[TreePath, Any]] = []
    for path, leaf in zip(paths, leaves, strict=True):
        try:
            normalized.append(_normalize_output_leaf(leaf, namespace=namespace))
        except TypeError:
            invalid.append((path, leaf))
    if invalid:
        labels = ", ".join(f"{format_path(path)} ({type(leaf).__name__})" for path, leaf in invalid)
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
    meta: _PrimitiveCallMeta,
    result: object,
    *,
    expected_input_count: int,
) -> tuple[object | None, ...]:
    """Flatten a structured transpose result into graph-input order."""
    if not isinstance(result, tuple):
        msg = "Primitive transpose rule must return a tuple of gradients"
        raise TypeError(msg)
    leaves, treedef = tree_flatten(result)
    if treedef == meta.call_treedef:
        mask = meta.input_leaf_mask
    else:
        args_mask, kwargs_mask = _unflatten_call_tree(
            meta.call_treedef,
            list(meta.input_leaf_mask),
        )
        kwargs_leaves, _ = tree_flatten(kwargs_mask)
        if any(kwargs_leaves):
            msg = (
                "Primitive transpose result must match the full (args, kwargs) "
                "pytree when keyword arguments contain traced values"
            )
            raise ValueError(msg)
        args_mask_leaves, args_treedef = tree_flatten(args_mask)
        if treedef != args_treedef:
            msg = "Primitive transpose result does not match the input pytree"
            raise ValueError(msg)
        mask = tuple(args_mask_leaves)
    gradients = tuple(leaf for is_input, leaf in zip(mask, leaves, strict=True) if is_input)
    if len(gradients) != expected_input_count:
        msg = (
            "Primitive transpose gradient count does not match node inputs: "
            f"expected {expected_input_count}, got {len(gradients)}"
        )
        raise TypeError(msg)
    return gradients


def _keyword_parameter(path: TreePath) -> str | None:
    if len(path) < _KEYWORD_PATH_LEN:
        return None
    root, parameter = path[0], path[1]
    if not isinstance(root, SequenceKey) or root.index != 1:
        return None
    if not isinstance(parameter, DictKey) or not isinstance(parameter.key, str):
        return None
    return parameter.key


def _trace_call_arguments(
    recorder: DynamicTape,
    *,
    op_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nondiff_argnames: frozenset[str],
    dynamic_argnames: frozenset[str],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[Any, ...],
    TreeDef,
    tuple[bool, ...],
    tuple[Any, ...],
    tuple[bool, ...],
    tuple[Any, ...],
    dict[str, Any],
    Any | None,
]:
    paths, leaves, treedef = tree_flatten_with_paths((args, kwargs))
    node_ids: list[int] = []
    parent_positions: list[int] = []
    literals: list[Any] = []
    input_mask: list[bool] = []
    static_leaves: list[Any] = []
    nondiff_mask: list[bool] = []
    call_leaves: list[Any] = []
    input_values: list[Any] = []
    namespace_values: list[Any] = []

    for path, leaf in zip(paths, leaves, strict=True):
        parameter = _keyword_parameter(path)
        dynamic_operand = _leaf_to_dynamic_operand(recorder, leaf)
        if dynamic_operand is not None:
            node_id, value = dynamic_operand
            operand_position = len(input_values)
            if node_id is None:
                literals.append(value)
            else:
                node_ids.append(node_id)
                parent_positions.append(operand_position)
            input_mask.append(True)
            nondiff_mask.append(parameter in nondiff_argnames)
            call_leaves.append(value)
            input_values.append(value)
            namespace_values.append(leaf)
            continue
        if parameter in dynamic_argnames:
            msg = (
                f"Primitive '{op_name.removeprefix('custom.')}' argument '{parameter}' "
                "is not traceable; declare it in static_argnames or pass an array/scalar"
            )
            raise TypeError(msg)
        input_mask.append(False)
        static_leaves.append(leaf)
        call_leaves.append(leaf)

    call_args, call_kwargs = _unflatten_call_tree(treedef, call_leaves)
    return (
        tuple(node_ids),
        tuple(parent_positions),
        tuple(literals),
        treedef,
        tuple(input_mask),
        tuple(static_leaves),
        tuple(nondiff_mask),
        call_args,
        call_kwargs,
        _infer_namespace(namespace_values),
    )


def _output_shape_and_dtype(value: Any) -> tuple[tuple[int, ...], Any]:
    if isinstance(value, ArrayLike):
        return tuple(int(size) for size in value.shape), value.dtype
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
    if op_def.num_outputs == count:
        return
    if op_def.num_outputs != 1:
        msg = (
            f"Primitive '{op_name.removeprefix('custom.')}' changed its output count "
            f"from {op_def.num_outputs} to {count}"
        )
        raise ValueError(msg)
    registry.update_num_outputs(op_name, num_outputs=count)


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


def _record_primitive_node(  # noqa: PLR0913 - mirrors the native recording contract
    recorder: DynamicTape,
    *,
    op: str,
    parents: tuple[int, ...],
    parent_positions: tuple[int, ...],
    literals: tuple[Any, ...],
    value: Any,
    attrs: Mapping[str, Any],
    shape: tuple[int, ...],
    dtype: Any,
    schema_version: int,
    source_location: str | None,
) -> int:
    if literals:
        return recorder.record_operation_with_literals(
            op,
            parents,
            parent_positions,
            literals,
            value,
            dict(attrs),
            shape,
            dtype,
            schema_version=schema_version,
            source_location=source_location,
            literal_weak=literals_are_weak(list(literals)),
        )
    return recorder.record_operation(
        op,
        parents,
        value,
        dict(attrs),
        shape,
        dtype,
        schema_version=schema_version,
        source_location=source_location,
    )


def _outer_tracer_recorder(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    current_recorder: object,
) -> object | None:
    """Return the nearest enclosing recorder represented by primitive operands."""
    leaves, _treedef = tree_flatten((args, kwargs))
    recorders = tuple(
        recorder
        for leaf in leaves
        if _is_traced_leaf(leaf)
        and (recorder := getattr(leaf, "recorder", None)) is not None
        and recorder is not current_recorder
    )
    if not recorders:
        return None
    return _select_deepest_active_recorder(recorders)


def trace_primitive_call(  # noqa: PLR0913 - one call carries the complete primitive contract
    function: Callable[..., Any],
    *,
    abstract_function: Callable[..., Any] | None,
    op_name: str,
    schema_version: int,
    recorder: DynamicTape,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    node_attrs: Mapping[str, Any],
    nondiff_argnames: frozenset[str],
    dynamic_argnames: frozenset[str],
    has_residual: bool,
    track_output_arity: bool = True,
) -> Any:
    """Execute one concrete primitive call and append its atomic tape node."""
    if _PRIMITIVE_CALL_KEY in kwargs:
        msg = f"Keyword argument '{_PRIMITIVE_CALL_KEY}' is reserved for Advect internals"
        raise TypeError(msg)
    (
        input_node_ids,
        input_positions,
        literals,
        call_treedef,
        input_leaf_mask,
        static_leaves,
        nondiff_input_mask,
        call_args,
        call_kwargs,
        namespace,
    ) = _trace_call_arguments(
        recorder,
        op_name=op_name,
        args=args,
        kwargs=kwargs,
        nondiff_argnames=nondiff_argnames,
        dynamic_argnames=dynamic_argnames,
    )
    outer_recorder = _outer_tracer_recorder(
        call_args,
        call_kwargs,
        current_recorder=recorder,
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
            execution = function(*call_args, **call_kwargs)
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
            nested_output = abstract_function(*call_args, **call_kwargs)
        else:
            nested_output = trace_primitive_call(
                function,
                abstract_function=abstract_function,
                op_name=op_name,
                schema_version=schema_version,
                recorder=cast("DynamicTape", outer_recorder),
                args=call_args,
                kwargs=call_kwargs,
                node_attrs=node_attrs,
                nondiff_argnames=nondiff_argnames,
                dynamic_argnames=dynamic_argnames,
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
        result_leaves, output_treedef = _normalize_output_pytree(
            execution.output,
            namespace=namespace,
        )
        meta = _PrimitiveCallMeta(
            call_treedef=call_treedef,
            input_leaf_mask=input_leaf_mask,
            static_leaves=static_leaves,
            output_treedef=output_treedef,
            nondiff_input_mask=nondiff_input_mask,
        )
        attrs = dict(node_attrs)
        attrs[_PRIMITIVE_CALL_KEY] = meta
        source_location = get_source_location()
        if track_output_arity:
            _record_primitive_output_count(op_name, len(result_leaves))

        if len(result_leaves) == 1:
            result_value = result_leaves[0]
            shape, dtype = _output_shape_and_dtype(result_value)
            node_id = _record_primitive_node(
                recorder,
                op=op_name,
                parents=input_node_ids,
                parent_positions=input_positions,
                literals=literals,
                value=result_value,
                attrs=attrs,
                shape=shape,
                dtype=dtype,
                schema_version=schema_version,
                source_location=source_location,
            )
            traced = _wrap_traced_output(
                result_value,
                node_id=node_id,
                recorder=recorder,
                namespace=namespace,
            )
            _attach_residual(recorder, node_id, execution)
            return tree_unflatten(output_treedef, [traced])

        shapes_dtypes = [_output_shape_and_dtype(leaf) for leaf in result_leaves]
        node_id = _record_primitive_node(
            recorder,
            op=op_name,
            parents=input_node_ids,
            parent_positions=input_positions,
            literals=literals,
            value=tuple(result_leaves),
            attrs=attrs,
            shape=shapes_dtypes[0][0],
            dtype=shapes_dtypes[0][1],
            schema_version=schema_version,
            source_location=source_location,
        )
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
