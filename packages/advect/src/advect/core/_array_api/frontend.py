# ruff: noqa: ANN401, FBT001, PLW1641
# ANN401: the Array API protocol is intentionally backend-agnostic.
# FBT001: __array__ follows NumPy's positional copy protocol.
# PLW1641: elementwise equality intentionally makes tracers unhashable.
"""Backend-neutral tracing for Python Array API implementations.

This frontend is deliberately small and provider-agnostic.  It recognizes an
Array API namespace from ``__array_namespace__``, evaluates operations through
that namespace, and records canonical ``array.*`` nodes.  NumPy has a richer,
separate frontend and is intentionally excluded here.

Operation binding remains separate from concrete provider execution so staging
can reuse the canonical call schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract_domains import operation_semantics
from advect.core._abstract_helpers import (
    accumulation_dtype,
    broadcast_shape as _broadcast_shape,
    dtype_name,
    normalize_axis as _normalize_axis,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)
from advect.core._array_api.providers import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_api.results import restore_array_api_result
from advect.core._array_api.signatures import (
    OFFICIAL_SIGNATURES,
    official_parameter_names,
    official_positional_parameter_names,
)
from advect.core._array_family_ops import _canonical_array_family_op_name
from advect.core._array_protocol_helpers import (
    literal_is_weak,
    literals_are_weak,
    materialize_weak_scalar_operands,
    normalize_item_index,
    weak_scalar_runtime_value,
)
from advect.core._context import (
    _get_active_array_api_version,
    _get_active_recorder,
    _is_recorder_in_active_trace_stack,
    _select_deepest_active_recorder,
    get_source_location,
    is_debug,
)
from advect.core._diagnostics import summarize_value
from advect.core._errors import (
    EscapedTracerError,
    MutationError,
    TracingError,
    _array_conversion_error,
)
from advect.core._protocols import _snapshot_traced
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from advect.core._native import DynamicTape

__all__ = [
    "ArrayAPICallBinding",
    "ArrayAPINamespace",
    "ArrayAPITracer",
    "bind_array_api_call",
]

_MIN_MATRIX_RANK = 2
_EXPAND_DIMS_POSITIONAL_ARITY = 2

_ARRAY_API_COMPOSITE_OPERANDS = {
    "broadcast_arrays": ("arrays",),
    "linalg.matrix_power": ("x",),
    "linalg.matrix_rank": ("x", "rtol"),
    "meshgrid": ("arrays",),
    "nonzero": ("x",),
    "unique_all": ("x",),
    "unique_counts": ("x",),
    "unique_inverse": ("x",),
    "unique_values": ("x",),
    "unstack": ("x",),
}
_ARRAY_API_COMPOSITES = frozenset(_ARRAY_API_COMPOSITE_OPERANDS)
_STAGED_ARRAY_API_COMPOSITES = frozenset(
    {
        "broadcast_arrays",
        "linalg.matrix_power",
        "linalg.matrix_rank",
        "meshgrid",
        "unstack",
    }
)
_DYNAMIC_ARRAY_API_COMPOSITES = _ARRAY_API_COMPOSITES - _STAGED_ARRAY_API_COMPOSITES
_NONDIFFERENTIABLE_ARRAY_API_COMPOSITES = frozenset(
    {
        "linalg.matrix_rank",
        "nonzero",
    }
)


@dataclass(frozen=True, slots=True)
class _FunctionSpec:
    op: str
    operands: tuple[str, ...]
    sequence_operands: frozenset[str] = frozenset()
    positional_operands: tuple[int, ...] = ()
    positional_attrs: tuple[tuple[int, str], ...] = ()


def _metadata_functions() -> frozenset[str]:
    return frozenset({"can_cast", "finfo", "iinfo", "isdtype", "result_type"})


def _function_specs() -> dict[str, _FunctionSpec]:
    unsupported = {
        "from_dlpack",
        *_ARRAY_API_COMPOSITES,
    }
    aliases = {
        "abs": "array.absolute",
        "acos": "array.arccos",
        "acosh": "array.arccosh",
        "asin": "array.arcsin",
        "asinh": "array.arcsinh",
        "asarray": "array.astype",
        "atan": "array.arctan",
        "atan2": "array.arctan2",
        "atanh": "array.arctanh",
        "bitwise_invert": "array.invert",
        "bitwise_left_shift": "array.left_shift",
        "bitwise_right_shift": "array.right_shift",
        "concat": "array.concatenate",
        "conj": "array.conjugate",
        "cumulative_prod": "array.cumprod",
        "cumulative_sum": "array.cumsum",
        "linalg.cross": "array.cross",
        "linalg.diagonal": "array.diagonal",
        "linalg.matmul": "array.matmul",
        "linalg.matrix_transpose": "array.transpose",
        "linalg.outer": "array.outer",
        "linalg.tensordot": "array.tensordot",
        "linalg.trace": "array.trace",
        "linalg.vecdot": "array.vecdot",
        "matrix_transpose": "array.transpose",
        "permute_dims": "array.transpose",
        "pow": "array.power",
        "round": "array.rint",
    }
    operand_exceptions = {
        "clip": ("x", "min", "max"),
        "full": ("fill_value",),
        "linalg.pinv": ("x", "rtol"),
    }
    schemas = {name: schema for name, schema, _evaluator in operation_semantics()}
    specs: dict[str, _FunctionSpec] = {}
    for path in OFFICIAL_SIGNATURES:
        if path in unsupported or path in _metadata_functions():
            continue
        suffix = path
        default_op = _canonical_array_family_op_name(suffix)
        op = aliases.get(path, default_op)
        schema = schemas[op]
        parameters = official_parameter_names(path)
        positional = official_positional_parameter_names(path)
        operands = operand_exceptions.get(path, parameters[: schema.operands])
        positional_operands = tuple(
            parameters.index(name) for name in operands if name in positional
        )
        positional_attrs = tuple(
            (parameters.index(name), name) for name in positional if name not in operands
        )
        specs[path] = _FunctionSpec(
            op=op,
            operands=operands,
            sequence_operands=(frozenset(operands[:1]) if schema.sequence_operand else frozenset()),
            positional_operands=positional_operands,
            positional_attrs=positional_attrs,
        )
    return specs


_FUNCTION_SPECS = _function_specs()

# Private provider extensions used while tracing derivative rules. Keeping
# these out of `_FUNCTION_SPECS` avoids advertising them as Array API surface
# or requiring providers which implement only the standard to expose them.
_INTERNAL_FUNCTION_SPECS: dict[str, _FunctionSpec] = {
    # Derivative rules use NumPy's historical spellings even when the selected
    # Array API revision predates the standard cumulative-function names.
    "cumprod": _FUNCTION_SPECS["cumulative_prod"],
    "cumsum": _FUNCTION_SPECS["cumulative_sum"],
    "ldexp": _FunctionSpec(
        "array_ext.ldexp",
        ("x", "exponent"),
        positional_operands=(0, 1),
    ),
    "linalg.eig": _FunctionSpec(
        "array_ext.linalg.eig",
        ("x",),
        positional_operands=(0,),
    ),
}
_BINARY_ARITY = 2
_ARRAY_API_META_FUNCTIONS = _metadata_functions()
_ACCUMULATION_FUNCTIONS = frozenset({"prod", "sum"})


@dataclass(frozen=True, slots=True)
class ArrayAPICallBinding:
    """Backend-independent result of binding one supported Array API call."""

    op: str
    operands: tuple[Any, ...]
    attrs: Mapping[str, Any]
    num_outputs: int


def _bind_array_api_arguments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    spec: _FunctionSpec,
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, str]]:
    bound = {f"__arg_{index}": value for index, value in enumerate(args)}
    bound.update(kwargs)
    aliases: dict[str, str] = {}
    positional_names: list[str] = []
    for operand_index, position in enumerate(spec.positional_operands):
        positional_name = f"__arg_{position}"
        positional_names.append(positional_name)
        if operand_index < len(spec.operands):
            aliases[positional_name] = spec.operands[operand_index]
    operand_names = (*positional_names, *(name for name in spec.operands if name in bound))
    return bound, operand_names, aliases


def _move_positional_array_api_attrs(
    path: str,
    spec: _FunctionSpec,
    attrs: dict[str, Any],
) -> None:
    for position, name in spec.positional_attrs:
        positional_name = f"__arg_{position}"
        if positional_name not in attrs:
            continue
        if name in attrs:
            msg = f"Array API {path}() received {name!r} twice"
            raise TypeError(msg)
        attrs[name] = attrs.pop(positional_name)


def _collect_array_api_operands(
    path: str,
    spec: _FunctionSpec,
    bound: dict[str, Any],
    operand_names: tuple[str, ...],
    operand_aliases: dict[str, str],
    attrs: dict[str, Any],
) -> list[Any]:
    operands: list[Any] = []
    for operand_name in operand_names:
        if operand_name not in bound:
            continue
        value = bound[operand_name]
        attrs.pop(operand_name, None)
        canonical_name = operand_aliases.get(operand_name, operand_name)
        if path == "clip" and canonical_name in {"min", "max"}:
            attrs[f"_advect_clip_{canonical_name}_is_input"] = value is not None
            if value is None:
                continue
        if path == "linalg.pinv" and canonical_name == "rtol":
            attrs["_advect_pinv_tolerance"] = "rtol" if value is not None else None
            if value is None:
                continue
        if canonical_name not in spec.sequence_operands:
            operands.append(value)
            continue
        if not isinstance(value, (tuple, list)):
            msg = f"Array API {path}() expects {canonical_name!r} to be a list or tuple"
            raise TypeError(msg)
        operands.extend(value)
    return operands


def _normalize_array_api_attrs(
    path: str,
    attrs: dict[str, Any],
    operands: list[Any],
) -> None:
    if path == "asarray":
        attrs["_advect_array_api_asarray"] = True
    device = attrs.pop("device", None)
    if device is not None:
        attrs["_advect_device"] = str(device)
    if path == "clip":
        attrs.setdefault("_advect_clip_min_is_input", False)
        attrs.setdefault("_advect_clip_max_is_input", False)
    if path == "tile" and "repetitions" in attrs:
        attrs["reps"] = attrs.pop("repetitions")
    if path == "sort":
        attrs.setdefault("descending", False)
        attrs.setdefault("stable", True)
    if path in {"linalg.diagonal", "linalg.trace"}:
        attrs["axis1"] = -2
        attrs["axis2"] = -1
    if not path.endswith("matrix_transpose") or not operands:
        return
    shape = getattr(operands[0], "shape", ())
    rank = len(shape)
    if rank < _MIN_MATRIX_RANK:
        msg = "matrix_transpose requires an array with at least two dimensions"
        raise ValueError(msg)
    axes = list(range(rank))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    attrs["axes"] = tuple(axes)


def bind_array_api_call(
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayAPICallBinding:
    """Classify data operands and static attributes without executing the call."""
    spec = _FUNCTION_SPECS.get(path) or _INTERNAL_FUNCTION_SPECS.get(path)
    if spec is None:
        msg = (
            f"Array API function {path!r} is not traceable yet. "
            "Define it as an Advect primitive or use supported Array API operations."
        )
        raise NotImplementedError(msg)

    bound, operand_names, operand_aliases = _bind_array_api_arguments(
        args,
        kwargs,
        spec,
    )
    attrs = dict(bound)
    _move_positional_array_api_attrs(path, spec, attrs)
    operands = _collect_array_api_operands(
        path,
        spec,
        bound,
        operand_names,
        operand_aliases,
        attrs,
    )
    _normalize_array_api_attrs(path, attrs, operands)

    return ArrayAPICallBinding(
        op=spec.op,
        operands=tuple(operands),
        attrs=attrs,
        num_outputs=get_registry().get(spec.op).num_outputs,
    )


def _normalize_provider_call(
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Adapt observed provider signatures to the selected official contract."""
    if path != "expand_dims" or len(args) != _EXPAND_DIMS_POSITIONAL_ARITY or "axis" in kwargs:
        return args, kwargs
    # All supported specifications accept ``axis`` positionally, while the
    # array-api-strict reference provider currently exposes it keyword-only.
    options = dict(kwargs)
    options["axis"] = args[1]
    return args[:1], options


def _normalize_accumulation_dtype(
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    namespace: Any,
    array_api_version: str,
) -> dict[str, Any]:
    """Apply the selected revision's default accumulation dtype."""
    if path not in _ACCUMULATION_FUNCTIONS or not args or kwargs.get("dtype") is not None:
        return kwargs
    source_dtype = args[0].dtype
    target_name = accumulation_dtype(
        source_dtype,
        array_api_version=array_api_version,
    )
    if target_name == dtype_name(source_dtype):
        return kwargs
    target_dtype = getattr(namespace, target_name, None)
    if target_dtype is None:
        msg = f"The runtime array namespace does not provide dtype {target_name!r}"
        raise TypeError(msg)
    normalized = dict(kwargs)
    normalized["dtype"] = target_dtype
    return normalized


def _selected_array_api_version() -> str:
    return _get_active_array_api_version() or LATEST_ARRAY_API_VERSION


def _raw_namespace(value: Any, *, api_version: str | None = None) -> Any | None:
    selected = _selected_array_api_version() if api_version is None else api_version
    resolved = _get_array_namespace(value, api_version=selected)
    if resolved is not None:
        return resolved
    namespace_function = getattr(value, "__array_namespace__", None)
    if not callable(namespace_function):
        return None
    try:
        return namespace_function(api_version=selected)
    except Exception:  # noqa: BLE001 - backend discovery must be non-invasive
        try:
            # A default namespace still lets the acceptance predicate emit a
            # precise version error for providers that cannot serve the pin.
            return namespace_function()
        except Exception:  # noqa: BLE001 - backend discovery must be non-invasive
            return None


def _is_standard_array_api_namespace(namespace: Any, *, api_version: str) -> bool:
    version = getattr(namespace, "__array_api_version__", None)
    namespace_info = getattr(namespace, "__array_namespace_info__", None)
    selected = tuple(int(part) for part in api_version.split("."))
    reported = (
        tuple(int(part) for part in version.split("."))
        if isinstance(version, str) and all(part.isdigit() for part in version.split("."))
        else None
    )
    return (
        reported is not None
        and reported >= selected
        and (api_version == "2022.12" or callable(namespace_info))
        and callable(getattr(namespace, "asarray", None))
    )


def _accepts_array_api(value: Any) -> bool:
    if isinstance(value, ArrayAPITracer):
        return True
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return False
    selected = _selected_array_api_version()
    namespace = _raw_namespace(value, api_version=selected)
    if namespace is None:
        return False
    # NumPy and its nested tracers retain their richer first-class frontend.
    backend = _get_backend_key_from_namespace(namespace)
    if backend is not None and backend.split(".", 1)[0] == "numpy":
        return False
    # Nested transforms legitimately present an enclosing tracer as the
    # "concrete" value of an inner trace. Its namespace has already been
    # validated by the enclosing frontend and intentionally exposes only the
    # operations admitted by that trace.
    if callable(getattr(value, "_advect_snapshot", None)):
        return True
    version = getattr(namespace, "__array_api_version__", None)
    namespace_info = getattr(namespace, "__array_namespace_info__", None)
    asarray = getattr(namespace, "asarray", None)
    if isinstance(version, str) and callable(namespace_info) and callable(asarray):
        if not _is_standard_array_api_namespace(namespace, api_version=selected):
            msg = f"Advect selected Array API {selected}; the input provider exposes {version}"
            raise TypeError(msg)
    elif not _is_standard_array_api_namespace(namespace, api_version=selected):
        return False
    return backend is not None


def _handle_array_api_input(
    value: Any,
    name: str | None = None,
    *,
    active: bool = True,
) -> ArrayAPITracer:
    recorder = _get_active_recorder()
    if recorder is None:
        msg = "Array API inputs require an active trace"
        raise RuntimeError(msg)
    selected = _selected_array_api_version()
    namespace = _raw_namespace(value, api_version=selected)
    if namespace is None:
        msg = f"{type(value).__name__} no longer exposes an Array API namespace"
        raise TypeError(msg)
    if not callable(getattr(value, "_advect_snapshot", None)) and not (
        _is_standard_array_api_namespace(namespace, api_version=selected)
    ):
        version = getattr(namespace, "__array_api_version__", None)
        msg = f"Advect selected Array API {selected}; the input provider exposes {version!r}"
        raise TypeError(msg)
    node_id = recorder.record_input(
        value,
        tuple(int(dimension) for dimension in value.shape),
        value.dtype,
        name=name,
        active=active,
    )
    return ArrayAPITracer(
        value,
        node_id,
        recorder,
        namespace=namespace,
        array_api_version=selected,
        owned=False,
    )


def _wrap_traced(value: Any, *, node_id: int, recorder: DynamicTape) -> ArrayAPITracer:
    selected = _selected_array_api_version()
    namespace = _raw_namespace(value, api_version=selected)
    if namespace is None:
        msg = "A traced Array API result no longer exposes its namespace"
        raise TypeError(msg)
    return ArrayAPITracer(
        value,
        node_id,
        recorder,
        namespace=namespace,
        array_api_version=selected,
    )


def _copy_array_value(value: Any, namespace: Any) -> Any:
    """Copy one provider value without changing its trace identity."""
    copy_value = getattr(value, "copy", None)
    if callable(copy_value):
        return copy_value()
    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        msg = "The Array API provider does not implement asarray(copy=True)"
        raise TypeError(msg)
    return asarray(value, copy=True)


def _unwrap(value: Any) -> Any:
    if isinstance(value, ArrayAPITracer):
        payload = _unwrap(_snapshot_traced(value)[1])
        return weak_scalar_runtime_value(value, payload)
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    return value


def _tracer_recorders(value: Any) -> tuple[DynamicTape, ...]:
    recorders: list[DynamicTape] = []
    current = value
    while isinstance(current, ArrayAPITracer):
        current._require_active_recorder()  # noqa: SLF001 - internal chain validation
        recorders.append(current.recorder)
        current = _snapshot_traced(current)[1]
    return tuple(recorders)


def _operand_for_recorder(
    operand: Any,
    *,
    recorder: DynamicTape,
) -> tuple[int | None, Any]:
    original = operand
    current = operand
    original_level: int | None = None
    while isinstance(current, ArrayAPITracer):
        owner = current.recorder
        level, _frame_id = owner.runtime_trace_identity()
        if original_level is None:
            original_level = level
        node_id, value = _snapshot_traced(current)
        if owner is recorder:
            return node_id, weak_scalar_runtime_value(current, value)
        current = value

    if isinstance(original, ArrayAPITracer):
        recorder_level, _frame_id = recorder.runtime_trace_identity()
        if (
            original_level is not None
            and recorder_level is not None
            and original_level < recorder_level
        ):
            return None, original
        return None, _unwrap(original)

    namespace = _raw_namespace(operand)
    if (
        namespace is not None and hasattr(operand, "shape") and hasattr(operand, "dtype")
    ) or isinstance(operand, (bool, int, float, complex)):
        return None, operand
    msg = f"Unsupported dynamic Array API operand of type {type(operand).__name__}"
    raise TypeError(msg)


def _record_array_api_operation(
    *,
    recorder: DynamicTape,
    op: str,
    operands: tuple[Any, ...],
    value: Any,
    attrs: dict[str, Any],
    shape: tuple[int, ...],
    dtype: Any,
) -> int:
    projected = tuple(_operand_for_recorder(operand, recorder=recorder) for operand in operands)
    parents = tuple(node_id for node_id, _value in projected if node_id is not None)
    literals = tuple(item for node_id, item in projected if node_id is None)
    source_location = get_source_location()
    if not literals:
        return recorder.record_operation(
            op,
            parents,
            value,
            attrs,
            shape,
            dtype,
            source_location=source_location,
        )
    parent_positions = tuple(
        position for position, (node_id, _value) in enumerate(projected) if node_id is not None
    )
    return recorder.record_operation_with_literals(
        op,
        parents,
        parent_positions,
        literals,
        value,
        attrs,
        shape,
        dtype,
        source_location=source_location,
        literal_weak=literals_are_weak(list(literals)),
    )


def _normalize_array_api_outputs(
    path: str,
    result: Any,
    *,
    num_outputs: int,
) -> tuple[tuple[Any, ...], tuple[tuple[tuple[int, ...], Any], ...]]:
    """Validate one declared result contract and normalize public tuples."""
    if num_outputs == 1:
        outputs = (result,)
    else:
        if not isinstance(result, tuple):
            msg = (
                f"Array API function {path!r} must return a tuple of {num_outputs} "
                f"outputs, got {type(result).__name__}"
            )
            raise TypeError(msg)
        outputs = tuple(result)
        if len(outputs) != num_outputs:
            msg = (
                f"Array API function {path!r} returned {len(outputs)} outputs, "
                f"expected {num_outputs}"
            )
            raise ValueError(msg)

    metadata: list[tuple[tuple[int, ...], Any]] = []
    for index, output in enumerate(outputs):
        if not hasattr(output, "shape") or not hasattr(output, "dtype"):
            suffix = "" if num_outputs == 1 else f" at output {index}"
            msg = (
                f"Array API function {path!r} returned "
                f"{type(output).__name__}{suffix}; data-dependent-shape results "
                "are not traceable"
            )
            raise NotImplementedError(msg)
        metadata.append(
            (
                tuple(int(dimension) for dimension in output.shape),
                output.dtype,
            )
        )
    return outputs, tuple(metadata)


def _record_array_api_result(
    *,
    recorder: DynamicTape,
    op: str,
    operands: tuple[Any, ...],
    values: tuple[Any, ...],
    attrs: dict[str, Any],
    metadata: tuple[tuple[tuple[int, ...], Any], ...],
    namespace: Any,
    array_api_version: str,
) -> Any:
    """Record one fixed-arity result and return recorder-local wrappers."""
    parent_value: Any = values[0] if len(values) == 1 else tuple(values)
    first_shape, first_dtype = metadata[0]
    parent_id = _record_array_api_operation(
        recorder=recorder,
        op=op,
        operands=operands,
        value=parent_value,
        attrs=attrs,
        shape=first_shape,
        dtype=first_dtype,
    )
    if len(values) == 1:
        return ArrayAPITracer(
            parent_value,
            parent_id,
            recorder,
            namespace=namespace,
            array_api_version=array_api_version,
        )

    outputs: list[ArrayAPITracer] = []
    for index, (value, (shape, dtype)) in enumerate(zip(values, metadata, strict=True)):
        output_id = recorder.record_operation(
            "advect.getoutput",
            (parent_id,),
            value,
            {"index": index, "num_outputs": len(values)},
            shape,
            dtype,
        )
        outputs.append(
            ArrayAPITracer(
                value,
                output_id,
                recorder,
                namespace=namespace,
                array_api_version=array_api_version,
            )
        )
    return tuple(outputs)


def _recorder_trace_level(recorder: DynamicTape) -> int:
    level, _frame_id = recorder.runtime_trace_identity()
    if level is None:
        msg = "Array API operation recorder is not bound to an active trace level"
        raise TracingError(msg)
    return level


def _array_api_tracers(value: Any) -> tuple[ArrayAPITracer, ...]:
    if isinstance(value, ArrayAPITracer):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tracer for item in value for tracer in _array_api_tracers(item))
    if isinstance(value, dict):
        return tuple(tracer for item in value.values() for tracer in _array_api_tracers(item))
    return ()


def _same_namespace(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return _get_backend_key_from_namespace(left) == _get_backend_key_from_namespace(right)


def _matrix_power_composite(namespace: Any, matrix: Any, exponent: object) -> Any:
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        message = "matrix_power exponent must be a static integer"
        raise TypeError(message)
    shape = tuple(int(size) for size in matrix.shape)
    if len(shape) < _MIN_MATRIX_RANK or shape[-2] != shape[-1]:
        message = "matrix_power requires square matrices"
        raise ValueError(message)
    if not namespace.isdtype(matrix.dtype, ("real floating", "complex floating")):
        message = "matrix_power requires a floating-point array"
        raise TypeError(message)
    if exponent == 0:
        ones = namespace.ones_like(matrix)
        return namespace.triu(namespace.tril(ones))

    base = namespace.linalg.inv(matrix) if exponent < 0 else matrix
    remaining = abs(exponent)
    result = None
    while remaining:
        if remaining & 1:
            result = base if result is None else namespace.matmul(result, base)
        remaining >>= 1
        if remaining:
            base = namespace.matmul(base, base)
    if result is None:  # pragma: no cover - exponent zero returns above
        message = "matrix_power failed to produce a result"
        raise AssertionError(message)
    return result


def _matrix_rank_composite(namespace: Any, matrix: Any, rtol: Any | None) -> Any:
    shape = tuple(int(size) for size in matrix.shape)
    if len(shape) < _MIN_MATRIX_RANK:
        message = "matrix_rank requires an array with at least two dimensions"
        raise ValueError(message)
    singular_values = namespace.linalg.svdvals(matrix)
    maximum = namespace.max(singular_values, axis=-1, keepdims=True)
    if rtol is None:
        rtol = max(shape[-2:]) * namespace.finfo(matrix.dtype).eps
    if not (hasattr(rtol, "shape") and hasattr(rtol, "dtype")):
        rtol = namespace.asarray(rtol, dtype=singular_values.dtype)
    rtol = namespace.expand_dims(rtol, axis=-1)
    tolerance = namespace.multiply(maximum, rtol)
    rank_mask = namespace.greater(singular_values, tolerance)
    return namespace.sum(namespace.astype(rank_mask, namespace.int64), axis=-1)


def _staged_array_api_composite(  # noqa: C901, PLR0912 - explicit bounded surface
    path: str,
    namespace: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if path == "broadcast_arrays":
        if not args or kwargs:
            message = "broadcast_arrays() expects one or more arrays"
            raise TypeError(message)
        shapes = tuple(tuple(int(size) for size in value.shape) for value in args)
        shape = _broadcast_shape(*shapes)
        return [namespace.broadcast_to(value, shape) for value in args]

    if path == "linalg.matrix_power":
        if len(args) != _BINARY_ARITY or kwargs:
            message = "matrix_power() expects a matrix and a static integer exponent"
            raise TypeError(message)
        return _matrix_power_composite(namespace, args[0], args[1])

    if path == "linalg.matrix_rank":
        if len(args) != 1 or set(kwargs) - {"rtol"}:
            message = "matrix_rank() expects one array and optional keyword-only rtol"
            raise TypeError(message)
        return _matrix_rank_composite(namespace, args[0], kwargs.get("rtol"))

    if path == "meshgrid":
        if not args or set(kwargs) - {"indexing"}:
            message = "meshgrid() expects one or more arrays and optional indexing"
            raise TypeError(message)
        indexing = kwargs.get("indexing", "xy")
        if indexing not in {"ij", "xy"}:
            message = "meshgrid indexing must be 'ij' or 'xy'"
            raise ValueError(message)
        if any(len(value.shape) != 1 for value in args):
            message = "meshgrid inputs must be one-dimensional"
            raise ValueError(message)
        if any(value.dtype != args[0].dtype for value in args[1:]):
            message = "meshgrid inputs must all have the same dtype"
            raise ValueError(message)
        if any(value.device != args[0].device for value in args[1:]):
            message = "meshgrid inputs must all be on the same device"
            raise ValueError(message)
        sizes = tuple(int(value.shape[0]) for value in args)
        rank = len(args)
        axes = list(range(rank))
        if indexing == "xy" and rank >= _BINARY_ARITY:
            axes[0], axes[1] = axes[1], axes[0]
        target = tuple(sizes[index] for index in axes)
        outputs = []
        for position, value in enumerate(args):
            shape = tuple(
                sizes[position] if dimension == axes[position] else 1 for dimension in range(rank)
            )
            reshaped = namespace.reshape(value, shape)
            outputs.append(namespace.broadcast_to(reshaped, target))
        return outputs

    if path == "unstack":
        if len(args) != 1 or set(kwargs) - {"axis"}:
            message = "unstack() expects one array and optional keyword-only axis"
            raise TypeError(message)
        value = args[0]
        rank = len(value.shape)
        axis_value = kwargs.get("axis", 0)
        axis = _normalize_axis(axis_value, rank)
        return tuple(
            value[tuple(index if dimension == axis else slice(None) for dimension in range(rank))]
            for index in range(int(value.shape[axis]))
        )

    message = f"Unknown staged Array API composite {path!r}"
    raise AssertionError(message)


class ArrayAPINamespace:
    """Trace-aware proxy for one concrete Python Array API namespace."""

    __slots__ = (
        "_array_api_version",
        "_namespace",
        "_path",
        "_profile",
        "_root_namespace",
    )

    def __init__(
        self,
        namespace: Any,
        *,
        path: str = "",
        root_namespace: Any | None = None,
        array_api_version: str = LATEST_ARRAY_API_VERSION,
    ) -> None:
        self._namespace = namespace
        self._path = path
        self._root_namespace = namespace if root_namespace is None else root_namespace
        self._array_api_version = array_api_version
        self._profile = materialize_array_api_profile(array_api_version)

    @property
    def __name__(self) -> str:
        return cast("str", getattr(self._namespace, "__name__", type(self._namespace).__name__))

    @property
    def __array_api_version__(self) -> str:
        return self._array_api_version

    @property
    def raw_namespace(self) -> Any:
        """Return the provider namespace used for concrete execution."""
        root = self._root_namespace
        return root.raw_namespace if isinstance(root, ArrayAPINamespace) else root

    def __array_namespace_info__(self) -> Any:
        return self.raw_namespace.__array_namespace_info__()

    def _advect_materialize_constant(self, value: object, spec: Any) -> Any:
        """Lift a provider constant through a nested dynamic trace.

        An outer abstract namespace may own a durable constant; a concrete
        provider can use the value directly. In both cases the temporary
        ``advect.const`` node gives the inner dynamic tape a dispatch identity
        without making the constant a differentiated input.
        """
        if bool(getattr(type(value), "__advect_abstract_array__", False)):
            materialized = value
        else:
            raw_namespace = self.raw_namespace
            materialize = getattr(raw_namespace, "_advect_materialize_constant", None)
            materialized = materialize(value, spec) if callable(materialize) else value
            if materialized is NotImplemented:
                materialized = value
        recorder = _get_active_recorder()
        if recorder is None or not (
            hasattr(materialized, "shape") and hasattr(materialized, "dtype")
        ):
            return materialized
        constant_value = cast("Any", materialized)
        node_id = recorder.record_operation(
            "advect.const",
            (),
            constant_value,
            {},
            tuple(int(dimension) for dimension in constant_value.shape),
            constant_value.dtype,
        )
        raw_namespace = self.raw_namespace
        return ArrayAPITracer(
            constant_value,
            node_id,
            cast("DynamicTape", recorder),
            namespace=raw_namespace,
            array_api_version=self._array_api_version,
            owned=False,
        )

    def __getattr__(self, name: str) -> Any:
        path = f"{self._path}.{name}" if self._path else name
        known_path = path in OFFICIAL_SIGNATURES
        if known_path and not self._profile.admits(path):
            message = (
                f"Array API function {path!r} is not available in the selected "
                f"{self._array_api_version} revision"
            )
            raise AttributeError(message)
        value = getattr(self._namespace, name)
        if not self._path and name in {"fft", "linalg"}:
            return ArrayAPINamespace(
                value,
                path=path,
                root_namespace=self.raw_namespace,
                array_api_version=self._array_api_version,
            )
        if callable(value):
            if isinstance(self._namespace, ArrayAPINamespace):
                raw_owner = self._namespace
                while isinstance(raw_owner, ArrayAPINamespace):
                    raw_owner = raw_owner._namespace  # noqa: SLF001 - nested proxy unwrapping
                raw_function = getattr(raw_owner, name)
                traced_function = value

                @wraps(raw_function)
                def call_through_parent(*args: Any, **kwargs: Any) -> Any:
                    return traced_function(*args, **kwargs)

                value = call_through_parent
            return partial(self._call, path, value)
        return value

    def __dir__(self) -> list[str]:
        names = set(super().__dir__()) | set(dir(self._namespace))
        if self._path:
            names = {
                name
                for name in names
                if f"{self._path}.{name}" not in OFFICIAL_SIGNATURES
                or self._profile.admits(f"{self._path}.{name}")
            }
        else:
            names = {
                name
                for name in names
                if name not in OFFICIAL_SIGNATURES or self._profile.admits(name)
            }
        return sorted(names)

    def _cumulative_with_initial(
        self,
        path: str,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not args:
            msg = f"{path}() requires an array"
            raise TypeError(msg)
        source = args[0]
        axis_value = kwargs.get("axis")
        source_shape = tuple(int(size) for size in source.shape)
        if axis_value is None:
            if len(source_shape) != 1:
                msg = "cumulative operations require axis= for inputs with more than one dimension"
                raise ValueError(msg)
            axis = 0
        else:
            axis = int(axis_value)
            if axis < 0:
                axis += len(source_shape)
            if axis < 0 or axis >= len(source_shape):
                msg = f"axis {axis_value} is out of bounds"
                raise ValueError(msg)
        base_kwargs = dict(kwargs)
        base_kwargs["axis"] = axis
        base_kwargs["include_initial"] = False
        base = self._call(path, function, *args, **base_kwargs)
        seed_shape = list(base.shape)
        seed_shape[axis] = 1
        fill_value = 1 if path == "cumulative_prod" else 0
        seed = self.raw_namespace.full(
            tuple(seed_shape),
            fill_value,
            dtype=base.dtype,
        )
        concat = cast("Callable[..., Any]", self.raw_namespace.concat)
        return self._call("concat", concat, (seed, base), axis=axis)

    def _diff_with_boundaries(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if len(args) != 1:
            message = "diff() expects one positional array argument"
            raise TypeError(message)
        options = dict(kwargs)
        prepend = options.pop("prepend", None)
        append = options.pop("append", None)
        if options.get("n", 1) == 0:
            return self._call("diff", function, args[0], **options)
        axis = options.get("axis", -1)
        parts = tuple(part for part in (prepend, args[0], append) if part is not None)
        concat = cast("Callable[..., Any]", self.raw_namespace.concat)
        joined = self._call("concat", concat, parts, axis=axis)
        return self._call("diff", function, joined, **options)

    def _searchsorted_with_sorter(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if len(args) != _BINARY_ARITY:
            message = "searchsorted() expects two positional array arguments"
            raise TypeError(message)
        options = dict(kwargs)
        sorter = options.pop("sorter")
        take = cast("Callable[..., Any]", self.raw_namespace.take)
        sorted_source = self._call("take", take, args[0], sorter, axis=0)
        return self._call("searchsorted", function, sorted_source, args[1], **options)

    def _asarray_live_sequence(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if len(args) != 1 or not isinstance(args[0], (tuple, list)):
            message = "asarray() expects one positional object argument"
            raise TypeError(message)
        if kwargs.get("copy") is False:
            message = "asarray(copy=False) cannot construct an array from a sequence"
            raise ValueError(message)

        def build(value: Any) -> Any:
            if isinstance(value, ArrayAPITracer):
                return value
            if isinstance(value, (tuple, list)):
                if not value:
                    return self.raw_namespace.asarray(value)
                children = tuple(build(item) for item in value)
                stack = cast("Callable[..., Any]", self.raw_namespace.stack)
                return self._call("stack", stack, children, axis=0)
            return self.raw_namespace.asarray(value)

        assembled = build(args[0])
        return self._call("asarray", function, assembled, **kwargs)

    def _composite_namespace(
        self,
        tracers: tuple[ArrayAPITracer, ...],
        *,
        path: str,
    ) -> ArrayAPINamespace:
        for tracer in tracers:
            tracer._require_active_recorder()  # noqa: SLF001 - same frontend invariant
            if not _same_namespace(tracer.raw_namespace, self.raw_namespace):
                message = f"Cannot combine different Array API namespaces in {path}()"
                raise TypeError(message)
        return ArrayAPINamespace(
            self.raw_namespace,
            array_api_version=self._array_api_version,
        )

    @staticmethod
    def _lift_discrete_composite(namespace: ArrayAPINamespace, value: Any) -> Any:
        return namespace._advect_materialize_constant(value, None)

    def _dynamic_array_api_composite(
        self,
        path: str,
        namespace: ArrayAPINamespace,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if len(args) != 1 or kwargs:
            message = f"{path}() expects one positional array argument"
            raise TypeError(message)
        source = args[0]
        raw_source = _unwrap(source)
        raw_function = getattr(self.raw_namespace, path)

        if path == "nonzero":
            result = raw_function(raw_source)
            return tuple(self._lift_discrete_composite(namespace, value) for value in result)

        all_result = self.raw_namespace.unique_all(raw_source)
        flattened = namespace.reshape(source, (-1,))
        values = namespace.take(flattened, all_result.indices, axis=0)
        if path == "unique_values":
            return values

        if path == "unique_all":
            result = all_result
            metadata = result[1:]
        elif path == "unique_counts":
            result = raw_function(raw_source)
            metadata = (all_result.counts,)
        elif path == "unique_inverse":
            result = raw_function(raw_source)
            metadata = (all_result.inverse_indices,)
        else:
            message = f"Unknown dynamic Array API composite {path!r}"
            raise AssertionError(message)
        return type(result)(
            values,
            *(self._lift_discrete_composite(namespace, value) for value in metadata),
        )

    def _call(  # noqa: C901, PLR0911 - one closed frontend dispatch
        self,
        path: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        args, kwargs = _normalize_provider_call(path, args, kwargs)
        kwargs = _normalize_accumulation_dtype(
            path,
            args,
            kwargs,
            namespace=self.raw_namespace,
            array_api_version=self._array_api_version,
        )
        if path in _ARRAY_API_META_FUNCTIONS:
            return function(
                *cast("tuple[Any, ...]", _unwrap(args)),
                **cast("dict[str, Any]", _unwrap(kwargs)),
            )
        call_tracers = (*_array_api_tracers(args), *_array_api_tracers(kwargs))
        if not call_tracers:
            return function(*args, **kwargs)
        if path in _ARRAY_API_COMPOSITES:
            namespace = self._composite_namespace(call_tracers, path=path)
            if path in _STAGED_ARRAY_API_COMPOSITES:
                return _staged_array_api_composite(path, namespace, args, kwargs)
            return self._dynamic_array_api_composite(path, namespace, args, kwargs)
        if path == "asarray" and isinstance(args[0] if args else None, (tuple, list)):
            return self._asarray_live_sequence(function, args, kwargs)
        if path == "diff" and any(kwargs.get(name) is not None for name in ("prepend", "append")):
            return self._diff_with_boundaries(function, args, kwargs)
        if path == "searchsorted" and kwargs.get("sorter") is not None:
            return self._searchsorted_with_sorter(function, args, kwargs)
        if path in {"cumulative_prod", "cumulative_sum"} and bool(
            kwargs.get("include_initial", False)
        ):
            return self._cumulative_with_initial(path, function, args, kwargs)
        binding = bind_array_api_call(path, args, kwargs)
        tracers = [operand for operand in binding.operands if isinstance(operand, ArrayAPITracer)]
        if not tracers:
            return function(*args, **kwargs)

        for tracer in tracers:
            tracer._require_active_recorder()  # noqa: SLF001 - same frontend invariant
            if not _same_namespace(tracer.raw_namespace, self.raw_namespace):
                msg = f"Cannot combine different Array API namespaces in {path}()"
                raise TypeError(msg)
        recorder = cast(
            "DynamicTape",
            _select_deepest_active_recorder(tracer.recorder for tracer in tracers),
        )

        concrete_args = cast("tuple[Any, ...]", _unwrap(args))
        concrete_kwargs = cast("dict[str, Any]", _unwrap(kwargs))
        concrete_args = cast(
            "tuple[Any, ...]",
            materialize_weak_scalar_operands(
                binding.op,
                concrete_args,
                namespace=self.raw_namespace,
            ),
        )
        result = function(*concrete_args, **concrete_kwargs)
        outputs, output_metadata = _normalize_array_api_outputs(
            path,
            result,
            num_outputs=binding.num_outputs,
        )

        attrs = dict(binding.attrs)
        root_namespace = self.raw_namespace
        backend = _get_backend_key_from_namespace(root_namespace)
        if backend is not None:
            attrs["_advect_backend"] = backend
        attrs["_advect_array_api_version"] = self.__array_api_version__

        result_value: Any = outputs
        recorder_chain = {
            nested_recorder
            for operand in binding.operands
            for nested_recorder in _tracer_recorders(operand)
        }
        target_level = _recorder_trace_level(recorder)
        enclosing_recorders = sorted(
            (
                nested_recorder
                for nested_recorder in recorder_chain
                if nested_recorder is not recorder
                and _recorder_trace_level(nested_recorder) < target_level
            ),
            key=_recorder_trace_level,
        )
        for target_recorder in (*enclosing_recorders, recorder):
            result_value = _record_array_api_result(
                recorder=target_recorder,
                op=binding.op,
                operands=binding.operands,
                values=cast("tuple[Any, ...]", result_value),
                attrs=attrs,
                metadata=output_metadata,
                namespace=root_namespace,
                array_api_version=self._array_api_version,
            )
            if binding.num_outputs == 1:
                result_value = (result_value,)
        if binding.num_outputs == 1:
            return cast("tuple[Any, ...]", result_value)[0]
        return restore_array_api_result(
            path,
            cast("tuple[Any, ...]", result_value),
        )


class ArrayAPITracer:
    """Concrete traced value for non-NumPy Python Array API arrays."""

    __slots__ = (
        "_array_api_version",
        "_namespace",
        "_namespace_proxy",
        "_node_id",
        "_owned",
        "_recorder",
        "_value",
    )

    __array_priority__ = 100_000
    __advect_namespace_is_instance_specific__ = True

    def __init__(
        self,
        value: Any,
        node_id: int,
        recorder: DynamicTape,
        *,
        namespace: Any,
        array_api_version: str,
        owned: bool = True,
    ) -> None:
        self._value = value
        self._node_id = node_id
        self._recorder = recorder
        self._namespace = namespace
        self._array_api_version = array_api_version
        self._namespace_proxy = ArrayAPINamespace(
            namespace,
            array_api_version=array_api_version,
        )
        self._owned = owned

    @property
    def recorder(self) -> DynamicTape:
        """Return the invocation recorder that owns this SSA value."""
        return self._recorder

    def _require_active_recorder(self) -> DynamicTape:
        if not _is_recorder_in_active_trace_stack(self._recorder):
            msg = (
                "This Array API tracer escaped the trace that created it. "
                "Return concrete values from the transform instead of retaining tracers."
            )
            raise EscapedTracerError(msg)
        return self._recorder

    @property
    def value(self) -> Any:
        """Reject public access to the trace-time payload."""
        msg = (
            "Tracer payloads are private Advect implementation details. "
            "Return a value from the traced function to materialize it."
        )
        raise TracingError(msg)

    def _advect_snapshot(self) -> tuple[int, Any]:
        """Return one internally validated SSA/value pair."""
        self._require_active_recorder()
        return self._node_id, self._value

    @property
    def node_id(self) -> int:
        self._require_active_recorder()
        return self._node_id

    @property
    def raw_namespace(self) -> Any:
        self._require_active_recorder()
        return self._namespace

    @property
    def shape(self) -> tuple[int, ...]:
        self._require_active_recorder()
        return tuple(int(dimension) for dimension in self._value.shape)

    @property
    def dtype(self) -> Any:
        self._require_active_recorder()
        return self._value.dtype

    @property
    def _advect_weak(self) -> bool:
        """Return the weak-scalar category of the current SSA value."""
        return bool(self._require_active_recorder().is_weak(self._node_id))

    def _advect_mark_weak(self) -> None:
        """Mark this rank-zero SSA value as a weak scalar."""
        self._require_active_recorder().mark_weak(self._node_id)

    def __repr__(self) -> str:
        """Return a compact trace identity or a bounded debug payload summary."""
        self._require_active_recorder()
        prefix = f"ArrayAPITracer(node=%{self._node_id}"
        if is_debug():
            return f"{prefix}, {summarize_value(self._value)})"
        return f"{prefix}, shape={self.shape}, dtype={self.dtype})"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def device(self) -> Any:
        self._require_active_recorder()
        return getattr(self._value, "device", None)

    def __array_namespace__(self, *, api_version: str | None = None) -> ArrayAPINamespace:
        self._require_active_recorder()
        current = self._array_api_version
        if api_version is not None and api_version != current:
            msg = f"Array API version {api_version!r} requested, but provider exposes {current!r}"
            raise ValueError(msg)
        return self._namespace_proxy

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Offer a foreign array protocol to its registered frontend."""
        from advect.core._backends import get_hook  # noqa: PLC0415 - avoid import cycle

        handler = get_hook("advect.foreign_array_ufunc")
        if handler is None:
            return NotImplemented
        return handler(self, ufunc, method, inputs, kwargs)

    def __array_function__(
        self,
        function: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Offer a foreign array-function protocol to its registered frontend."""
        del types
        from advect.core._backends import get_hook  # noqa: PLC0415 - avoid import cycle

        handler = get_hook("advect.foreign_array_function")
        if handler is None:
            return NotImplemented
        return handler(self, function, args, kwargs)

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> Any:
        del dtype, copy
        raise TracingError(_array_conversion_error())

    def __bool__(self) -> bool:
        self._require_active_recorder()
        return bool(self._value)

    def __len__(self) -> int:
        self._require_active_recorder()
        if not self.shape:
            msg = "len() of a 0-dimensional array"
            raise TypeError(msg)
        return self.shape[0]

    def __iter__(self) -> Any:
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: Any) -> ArrayAPITracer:
        recorder = self._require_active_recorder()
        result = self._value[index]
        node_id = recorder.record_operation(
            "advect.getitem",
            (self.node_id,),
            result,
            {
                "index": index,
                "_advect_backend": _get_backend_key_from_namespace(self._namespace),
            },
            tuple(int(dimension) for dimension in result.shape),
            result.dtype,
        )
        return ArrayAPITracer(
            result,
            node_id,
            recorder,
            namespace=self._namespace,
            array_api_version=self._array_api_version,
            owned=False,
        )

    def __setitem__(self, index: Any, value: Any) -> None:
        recorder = self._require_active_recorder()
        if not self._owned:
            msg = (
                "Mutation for generic Array API inputs is not supported. "
                "Copy the value before applying an internal functional update."
            )
            raise MutationError(msg)
        source_id, source_value = self._advect_snapshot()
        replacement_id, replacement_value = _operand_for_recorder(value, recorder=recorder)
        result = _copy_array_value(source_value, self._namespace)
        result[index] = replacement_value
        attrs = {"index": index, "mode": "set"}
        if replacement_id is None:
            node_id = recorder.record_operation_with_literals(
                "advect.index_update",
                (source_id,),
                (0,),
                (replacement_value,),
                result,
                attrs,
                tuple(int(dimension) for dimension in result.shape),
                result.dtype,
                literal_weak=literal_is_weak(value),
            )
        else:
            node_id = recorder.record_operation(
                "advect.index_update",
                (source_id, replacement_id),
                result,
                attrs,
                tuple(int(dimension) for dimension in result.shape),
                result.dtype,
            )
        self._node_id = node_id
        self._value = result

    def copy(self) -> ArrayAPITracer:
        """Record an owned copy for staged graph replay."""
        recorder = self._require_active_recorder()
        source_id, source_value = self._advect_snapshot()
        result = _copy_array_value(source_value, self._namespace)
        node_id = recorder.record_operation(
            "advect.copy",
            (source_id,),
            result,
            {},
            tuple(int(dimension) for dimension in result.shape),
            result.dtype,
        )
        return ArrayAPITracer(
            result,
            node_id,
            recorder,
            namespace=self._namespace,
            array_api_version=self._array_api_version,
            owned=True,
        )

    def astype(
        self,
        dtype: Any,
        *,
        copy: bool = True,
        device: Any | None = None,
    ) -> ArrayAPITracer:
        namespace = self.__array_namespace__()
        kwargs: dict[str, Any] = {"copy": copy}
        if device is not None:
            kwargs["device"] = device
        return cast("ArrayAPITracer", namespace.astype(self, dtype, **kwargs))

    def reshape(self, *shape: Any) -> ArrayAPITracer:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else tuple(shape)
        return cast("ArrayAPITracer", self.__array_namespace__().reshape(self, target))

    def item(self, *args: object) -> ArrayAPITracer:
        """Return one element as a rank-zero traced value."""
        index = normalize_item_index(args, ndim=self.ndim)
        if index is None:
            if self.size != 1:
                msg = "can only convert an array of size 1 to a scalar"
                raise ValueError(msg)
            if self.shape == ():
                return self
            return self[tuple(0 for _dimension in self.shape)]
        if isinstance(index, tuple):
            return self[index]
        return self.reshape((-1,))[index]

    def sum(self, *, axis: Any = None, dtype: Any = None, keepdims: bool = False) -> Any:
        kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self.__array_namespace__().sum(self, **kwargs)

    def conj(self) -> ArrayAPITracer:
        return cast("ArrayAPITracer", self.__array_namespace__().conj(self))

    @property
    def real(self) -> ArrayAPITracer:
        return cast("ArrayAPITracer", self.__array_namespace__().real(self))

    @property
    def imag(self) -> ArrayAPITracer:
        return cast("ArrayAPITracer", self.__array_namespace__().imag(self))

    @property
    def T(self) -> ArrayAPITracer:  # noqa: N802 - standard array spelling
        axes = tuple(reversed(range(self.ndim)))
        return cast("ArrayAPITracer", self.__array_namespace__().permute_dims(self, axes))

    @property
    def mT(self) -> ArrayAPITracer:  # noqa: N802 - standard array spelling
        return cast("ArrayAPITracer", self.__array_namespace__().matrix_transpose(self))

    def _binary(self, name: str, other: Any, *, reverse: bool = False) -> Any:
        if (
            bool(getattr(type(self._value), "__advect_abstract_array__", False))
            and getattr(type(other), "__advect_frontend__", None) is not None
        ):
            return NotImplemented
        namespace = self.__array_namespace__()
        function = getattr(namespace, name)
        return function(other, self) if reverse else function(self, other)

    def __add__(self, other: Any) -> Any:
        return self._binary("add", other)

    def __radd__(self, other: Any) -> Any:
        return self._binary("add", other, reverse=True)

    def __sub__(self, other: Any) -> Any:
        return self._binary("subtract", other)

    def __rsub__(self, other: Any) -> Any:
        return self._binary("subtract", other, reverse=True)

    def __mul__(self, other: Any) -> Any:
        return self._binary("multiply", other)

    def __rmul__(self, other: Any) -> Any:
        return self._binary("multiply", other, reverse=True)

    def __truediv__(self, other: Any) -> Any:
        return self._binary("divide", other)

    def __rtruediv__(self, other: Any) -> Any:
        return self._binary("divide", other, reverse=True)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary("floor_divide", other)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._binary("floor_divide", other, reverse=True)

    def __mod__(self, other: Any) -> Any:
        return self._binary("remainder", other)

    def __rmod__(self, other: Any) -> Any:
        return self._binary("remainder", other, reverse=True)

    def __pow__(self, other: Any) -> Any:
        return self._binary("pow", other)

    def __rpow__(self, other: Any) -> Any:
        return self._binary("pow", other, reverse=True)

    def __matmul__(self, other: Any) -> Any:
        return self._binary("matmul", other)

    def __rmatmul__(self, other: Any) -> Any:
        return self._binary("matmul", other, reverse=True)

    def __and__(self, other: Any) -> Any:
        return self._binary("bitwise_and", other)

    def __rand__(self, other: Any) -> Any:
        return self._binary("bitwise_and", other, reverse=True)

    def __or__(self, other: Any) -> Any:
        return self._binary("bitwise_or", other)

    def __ror__(self, other: Any) -> Any:
        return self._binary("bitwise_or", other, reverse=True)

    def __xor__(self, other: Any) -> Any:
        return self._binary("bitwise_xor", other)

    def __rxor__(self, other: Any) -> Any:
        return self._binary("bitwise_xor", other, reverse=True)

    def __lt__(self, other: Any) -> Any:
        return self._binary("less", other)

    def __le__(self, other: Any) -> Any:
        return self._binary("less_equal", other)

    def __eq__(self, other: object) -> Any:
        return self._binary("equal", other)

    def __ne__(self, other: object) -> Any:
        return self._binary("not_equal", other)

    def __gt__(self, other: Any) -> Any:
        return self._binary("greater", other)

    def __ge__(self, other: Any) -> Any:
        return self._binary("greater_equal", other)

    def __neg__(self) -> Any:
        return self.__array_namespace__().negative(self)

    def __pos__(self) -> Any:
        return self.__array_namespace__().positive(self)

    def __abs__(self) -> Any:
        return self.__array_namespace__().abs(self)

    def __invert__(self) -> Any:
        return self.__array_namespace__().bitwise_invert(self)
