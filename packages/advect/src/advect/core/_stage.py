# ruff: noqa: ANN401, SLF001
"""Orchestrate Python staging across abstract tracing and the durable runtime.

This module owns call-signature validation, snapshots static inputs, drives an
abstract trace, and exposes the resulting program and compilation diagnostics.
It consumes provider-neutral semantics from :mod:`advect.core._abstract` and
hands graph construction, optimization, storage, and execution to the native
Rust boundary. Primitive definitions and derivative rules remain outside this
module, and Python does not maintain a parallel durable graph model here.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from itertools import pairwise
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract import (
    AbstractArray,
    AbstractTrace,
    AbstractValue,
    ArraySpec,
    _append_node,
    _dtype_name,
    _lift,
    _new_abstract_array,
)
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    materialize_array_api_profile,
)
from advect.core._array_api.providers import (
    ResolvedArrayNamespace,
    _get_backend_key_from_namespace,
    _get_provider_array_api_version,
    _negotiate_array_namespace_for_call,
    _version_key,
)
from advect.core._backends import get_hook
from advect.core._context import _set_active_recorder
from advect.core._eval_dispatch import bind_native_node_evaluator
from advect.core._graph_attrs import decode_graph_attrs_from_native
from advect.core._native import (
    build_graph_execution_plan,
    create_graph_builder,
    deserialize_graph_json,
    execute_graph,
)
from advect.core._portable_constant import (
    _PortableConstant,
    iter_constant_values,
    normalize_constant_dtype,
    portable_constant_from_native,
    snapshot_constant_parts,
)
from advect.core._primitive_call import (
    _PRIMITIVE_CALL_KEY,
    _keyword_parameter,
    _PrimitiveCallMeta,
    _record_primitive_output_count,
    _split_primitive_attrs,
)
from advect.core._pytree import (
    _get_node_impl,
    tree_flatten,
    tree_flatten_with_paths,
    tree_unflatten,
)
from advect.core._registry import get_registry
from advect.core._stage_serialization import (
    _decode_treedef,
    _decode_value,
    _encode_treedef,
    _encode_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import FrameType

    from advect.core._native import GraphBuilder, GraphExecutionPlan, GraphStore
    from advect.core._pytree import TreeDef

_ADVECT_ARRAY_SEMANTIC_PROFILE = "advect-array-1"


@dataclass(frozen=True, slots=True)
class StaticSpec:
    """An explicit compile-time Python value in a staged call signature.

    Examples
    --------
    >>> import advect as ad
    >>> ad.StaticSpec("sum").value
    'sum'
    """

    value: Any


@dataclass(frozen=True, slots=True)
class ConstantRecord:
    """Inspectable provenance for one concrete value captured while staging.

    Attributes
    ----------
    value_id
        Identifier of the constant-producing node. Records on a
        ``StagedProgram`` use optimized graph numbering; records on a
        ``StagedTrace`` use pre-optimization tape numbering.
    origin
        Capture category: ``"closure"``, ``"global"``, or ``"created"``.
    location
        Source location associated with the capture, when available.
    shape
        Captured array shape.
    dtype
        Canonical dtype name stored in the durable artifact.
    bytes
        Number of bytes in the captured value payload.
    digest
        Content digest used to identify the captured value.
    name
        Source-level name associated with the capture, when available.
    """

    value_id: int
    origin: str
    location: str | None
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    digest: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class _MaterializedConstants:
    """Provider-local constants retained by one compiled artifact."""

    namespace: object | None
    device: str | None
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class OptimizationPass:
    """Diagnostics for one required pass in the staged compiler.

    Attributes
    ----------
    name
        Stable pass name.
    nodes_before
        Graph node count before the pass.
    nodes_after
        Graph node count after the pass.
    removed_nodes
        Number of input nodes that have no output representative.
    rewritten_nodes
        Number of input nodes removed or mapped to a different node.
    """

    name: str
    nodes_before: int
    nodes_after: int
    removed_nodes: int
    rewritten_nodes: int


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Inspectable result of the fixed staged optimization pipeline.

    Attributes
    ----------
    nodes_before
        Graph node count before the first required pass.
    nodes_after
        Graph node count after the final required pass.
    rewritten_nodes
        Total number of nodes rewritten across all passes.
    passes
        Ordered diagnostics for each required optimization pass.
    """

    nodes_before: int
    nodes_after: int
    rewritten_nodes: int
    passes: tuple[OptimizationPass, ...]


@dataclass(frozen=True, slots=True)
class TracedNode:
    """One pre-optimization tape entry captured while staging.

    Attributes
    ----------
    id
        Position of the value-producing entry in the staging tape.
    op
        Canonical registered operation identifier.
    inputs
        Tape identifiers consumed by the operation.
    name
        Source-level input or constant name, when available.
    """

    id: int
    op: str
    inputs: tuple[int, ...]
    name: str | None


@dataclass(frozen=True, slots=True)
class StagedTrace:
    """The staging tape before cleanup and its mapping onto the final graph.

    ``old_to_new[node.id]`` is the optimized node that carries the traced
    node's value, or ``None`` when the cleanup pipeline removed it. Several
    traced nodes mapping onto one optimized node were merged by cse.
    ``constants`` holds the captured constant records in tape numbering. The
    trace is an in-process staging byproduct: it is not serialized, and it is
    ``None`` on programs loaded from a durable artifact.
    """

    nodes: tuple[TracedNode, ...]
    old_to_new: tuple[int | None, ...]
    constants: tuple[ConstantRecord, ...]


@dataclass(frozen=True, slots=True)
class _CompiledStage:
    graph: GraphStore
    execution_plan: GraphExecutionPlan
    call_treedef: TreeDef
    call_specs: tuple[ArraySpec | StaticSpec, ...]
    output_treedef: TreeDef
    output_specs: tuple[ArraySpec, ...]
    constants: tuple[ConstantRecord, ...]
    optimization: OptimizationReport
    trace: StagedTrace | None = None


@dataclass(slots=True)
class _ExecutionState:
    """Provider-local caches that do not participate in program identity."""

    materialized_constants: list[_MaterializedConstants] = field(
        default_factory=list,
        repr=False,
    )
    materialization_lock: Any = field(
        default_factory=Lock,
        repr=False,
    )


_STAGED_PROGRAM_FORMAT = "advect.ssa-program"
_STAGED_PROGRAM_FORMAT_VERSION = 2
_OPTIMIZATION_PASS_NAMES = ("dce", "simplify", "cse")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _encode_optimization(report: OptimizationReport) -> dict[str, object]:
    return {
        "nodes_before": report.nodes_before,
        "nodes_after": report.nodes_after,
        "rewritten_nodes": report.rewritten_nodes,
        "passes": [
            {
                "name": item.name,
                "nodes_before": item.nodes_before,
                "nodes_after": item.nodes_after,
                "removed_nodes": item.removed_nodes,
                "rewritten_nodes": item.rewritten_nodes,
            }
            for item in report.passes
        ],
    }


def _decode_optimization(payload: object) -> OptimizationReport:
    if not isinstance(payload, dict):
        raise TypeError("Staged optimization report must be a mapping")
    if set(payload) != {"nodes_before", "nodes_after", "rewritten_nodes", "passes"}:
        raise ValueError("Staged optimization report has invalid fields")
    for key in ("nodes_before", "nodes_after", "rewritten_nodes"):
        value = payload[key]
        if type(value) is not int or value < 0:
            raise TypeError(f"Staged optimization {key} must be a non-negative integer")
    raw_passes = payload["passes"]
    if not isinstance(raw_passes, list):
        raise TypeError("Staged optimization passes must be a list")
    passes: list[OptimizationPass] = []
    for raw_pass in raw_passes:
        if not isinstance(raw_pass, dict):
            raise TypeError("Staged optimization pass must be a mapping")
        required = {
            "name",
            "nodes_before",
            "nodes_after",
            "removed_nodes",
            "rewritten_nodes",
        }
        if set(raw_pass) != required:
            raise ValueError("Staged optimization pass has invalid fields")
        name = raw_pass["name"]
        if not isinstance(name, str):
            raise TypeError("Staged optimization pass name must be a string")
        counts: dict[str, int] = {}
        for key in required - {"name"}:
            value = raw_pass[key]
            if type(value) is not int or value < 0:
                raise TypeError(f"Staged optimization pass {key} must be a non-negative integer")
            counts[key] = value
        if counts["removed_nodes"] != counts["nodes_before"] - counts["nodes_after"]:
            raise ValueError("Staged optimization removed-node count is inconsistent")
        passes.append(
            OptimizationPass(
                name=name,
                nodes_before=counts["nodes_before"],
                nodes_after=counts["nodes_after"],
                removed_nodes=counts["removed_nodes"],
                rewritten_nodes=counts["rewritten_nodes"],
            )
        )
    if tuple(item.name for item in passes) != _OPTIMIZATION_PASS_NAMES:
        raise ValueError("Staged optimization pass sequence is invalid")
    nodes_before = cast("int", payload["nodes_before"])
    nodes_after = cast("int", payload["nodes_after"])
    rewritten_nodes = cast("int", payload["rewritten_nodes"])
    if (
        passes[0].nodes_before != nodes_before
        or passes[-1].nodes_after != nodes_after
        or any(left.nodes_after != right.nodes_before for left, right in pairwise(passes))
        or sum(item.rewritten_nodes for item in passes) != rewritten_nodes
    ):
        raise ValueError("Staged optimization aggregate counts are inconsistent")
    return OptimizationReport(
        nodes_before=nodes_before,
        nodes_after=nodes_after,
        rewritten_nodes=rewritten_nodes,
        passes=tuple(passes),
    )


def _encode_spec(spec: ArraySpec | StaticSpec) -> dict[str, object]:
    if isinstance(spec, ArraySpec):
        return {
            "kind": "array",
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "device": spec.device,
            "weak": spec.weak,
        }
    return {"kind": "static", "value": _encode_value(spec.value)}


def _decode_spec(payload: object) -> ArraySpec | StaticSpec:
    if not isinstance(payload, dict):
        raise TypeError("Staged call spec must be a mapping")
    kind = payload.get("kind")
    if kind == "array":
        if set(payload) != {"kind", "shape", "dtype", "device", "weak"}:
            raise ValueError("Staged array spec has invalid fields")
        shape = payload["shape"]
        dtype = payload["dtype"]
        device = payload["device"]
        weak = payload["weak"]
        if not isinstance(shape, list) or any(type(size) is not int for size in shape):
            raise TypeError("Staged array shape must be a list of integers")
        if not isinstance(dtype, str):
            raise TypeError("Staged array dtype must be a string")
        if device is not None and not isinstance(device, str):
            raise TypeError("Staged array device must be a string or None")
        if not isinstance(weak, bool):
            raise TypeError("Staged array weak flag must be a bool")
        return ArraySpec(tuple(shape), _dtype_name(dtype), device=device, weak=weak)
    if kind == "static":
        if set(payload) != {"kind", "value"}:
            raise ValueError("Staged static spec has invalid fields")
        return StaticSpec(_decode_value(payload["value"]))
    raise ValueError(f"Unknown staged call spec kind {kind!r}")


def _encode_constant(record: ConstantRecord) -> dict[str, object]:
    return {
        "value_id": record.value_id,
        "origin": record.origin,
        "location": record.location,
        "shape": list(record.shape),
        "dtype": record.dtype,
        "bytes": record.bytes,
        "digest": record.digest,
        "name": record.name,
    }


def _decode_constant(payload: object) -> ConstantRecord:
    if not isinstance(payload, dict):
        raise TypeError("Staged constant record must be a mapping")
    required = {"value_id", "origin", "location", "shape", "dtype", "bytes", "digest", "name"}
    if set(payload) != required:
        raise ValueError("Staged constant record has invalid fields")
    value_id = payload["value_id"]
    origin = payload["origin"]
    location = payload["location"]
    shape = payload["shape"]
    dtype = payload["dtype"]
    byte_count = payload["bytes"]
    digest = payload["digest"]
    name = payload["name"]
    if type(value_id) is not int or value_id < 0:
        raise TypeError("Staged constant value_id must be a non-negative integer")
    if origin not in {"closure", "global", "created"}:
        raise ValueError("Staged constant origin must be closure, global, or created")
    if location is not None and not isinstance(location, str):
        raise TypeError("Staged constant location must be a string or None")
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise TypeError("Staged constant shape must be a list of non-negative integers")
    if not isinstance(dtype, str) or not dtype:
        raise TypeError("Staged constant dtype must be a non-empty string")
    if type(byte_count) is not int or byte_count < 0:
        raise TypeError("Staged constant bytes must be a non-negative integer")
    if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
        raise TypeError("Staged constant digest must be a lowercase SHA-256 hex string")
    if name is not None and not isinstance(name, str):
        raise TypeError("Staged constant name must be a string or None")
    return ConstantRecord(
        value_id=value_id,
        origin=origin,
        location=location,
        shape=tuple(shape),
        dtype=dtype,
        bytes=byte_count,
        digest=digest,
        name=name,
    )


def _value_spec(value: Any) -> ArraySpec:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        device_value = getattr(value, "device", None)
        device = None if device_value is None else str(device_value)
        return ArraySpec(
            tuple(int(size) for size in shape),
            _dtype_name(dtype),
            device=device,
            weak=bool(getattr(value, "_advect_weak", False)),
        )
    if isinstance(value, bool):
        return ArraySpec((), "bool", weak=True)
    if isinstance(value, complex):
        return ArraySpec((), "complex128", weak=True)
    if isinstance(value, float):
        return ArraySpec((), "float64", weak=True)
    if isinstance(value, int):
        return ArraySpec((), "int64", weak=True)
    msg = (
        f"Staged array argument is not array-like: {type(value).__name__}; "
        "declare non-array inputs with StaticSpec(value)"
    )
    raise TypeError(msg)


def _specs_from_examples(examples: tuple[Any, ...]) -> tuple[Any, ...]:
    leaves, treedef = tree_flatten(examples)
    specs = [leaf if isinstance(leaf, StaticSpec) else _value_spec(leaf) for leaf in leaves]
    return cast("tuple[Any, ...]", tree_unflatten(treedef, specs))


def _normalize_weak_runtime_scalar(value: object, spec: ArraySpec) -> object:
    """Normalize one Python scalar to the exact declared weak dtype category."""
    dtype = spec.dtype
    if dtype == "bool":
        if type(value) is not bool:
            raise ValueError(
                f"Weak staged argument expected dtype={dtype}, got {type(value).__name__}"
            )
        return value
    if dtype.startswith("complex"):
        if type(value) is bool:
            raise ValueError(f"Weak staged argument expected dtype={dtype}, got bool")
        return complex(cast("Any", value))
    if dtype.startswith("float"):
        if type(value) not in {float, int}:
            raise ValueError(
                f"Weak staged argument expected a real scalar for dtype={dtype}, "
                f"got {type(value).__name__}"
            )
        return float(cast("Any", value))
    if dtype.startswith(("int", "uint")):
        if type(value) is not int:
            raise ValueError(
                f"Weak staged argument expected an integer for dtype={dtype}, "
                f"got {type(value).__name__}"
            )
        return value
    raise ValueError(f"Unsupported weak staged scalar dtype {spec.dtype!r}")


def _same_static_value(runtime: object, compiled: Any) -> bool:
    """Compare a runtime value with compiled static metadata by serialized identity.

    Python equality conflates ``1``, ``1.0``, and ``True`` (and ``0.0`` with
    ``-0.0``), each of which can trace a different graph. The codec encodes
    only exact builtin types, so values of different types never match.
    """
    kind = type(compiled)
    if type(runtime) is not kind:
        return False
    if kind is float:
        return repr(runtime) == repr(compiled)
    if kind is list or kind is tuple:
        return len(runtime) == len(compiled) and all(map(_same_static_value, runtime, compiled))
    if kind is dict:
        # Map each compiled key to the runtime's own equal key object.
        keys = {key: key for key in runtime}
        return len(keys) == len(compiled) and all(
            key in keys
            and _same_static_value(keys[key], key)
            and _same_static_value(runtime[key], item)
            for key, item in compiled.items()
        )
    return runtime == compiled


def _snapshot_static_value(value: object) -> object:
    """Return the closed, immutable-by-ownership value stored in an artifact spec."""
    return _decode_value(_encode_value(value))


def _flatten_runtime_to_treedef(value: Any, treedef: TreeDef) -> list[Any]:
    """Flatten a call according to its declared spec tree.

    A leaf in the spec tree stays a leaf even when its runtime value is itself a
    registered container. This is what makes ``StaticSpec(config_dict)`` an
    actual whole-value static declaration.
    """
    if treedef.node_type is None:
        return [value]
    if type(value) is not treedef.node_type:
        raise TypeError("Staged call pytree differs from the declared specs")
    implementation = _get_node_impl(treedef.node_type)
    if implementation is None:
        raise TypeError("Staged call uses an unregistered pytree node")
    flatten_fn, _unflatten_fn = implementation
    children, aux_data = flatten_fn(value)
    if treedef.node_type is dict:
        expected_keys = tuple(treedef.aux_data)
        entries = {key: (key, child) for key, child in zip(aux_data, children, strict=True)}
        if entries.keys() != set(expected_keys) or not all(
            _same_static_value(entries[key][0], key) for key in expected_keys
        ):
            raise TypeError("Staged call pytree differs from the declared specs")
        children = tuple(entries[key][1] for key in expected_keys)
    elif not _same_static_value(aux_data, treedef.aux_data):
        raise TypeError("Staged call pytree differs from the declared specs")
    if len(children) != len(treedef.children):
        raise TypeError("Staged call pytree differs from the declared specs")
    leaves: list[Any] = []
    for child, child_treedef in zip(children, treedef.children, strict=True):
        leaves.extend(_flatten_runtime_to_treedef(child, child_treedef))
    return leaves


def _capture_location() -> str | None:
    """Return the innermost caller frame outside the Advect package."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if not isinstance(module, str) or (module != "advect" and not module.startswith("advect.")):
            code = frame.f_code
            return f"{code.co_filename}:{frame.f_lineno} in {code.co_name}()"
        frame = frame.f_back
    return None


class _StageBuilder:
    __slots__ = (
        "_builder",
        "_closure_names",
        "_constant_ids",
        "_constant_values",
        "_global_names",
        "constants",
        "weak_constant_ids",
    )

    def __init__(self, function: Callable[..., Any], builder: GraphBuilder) -> None:
        self._builder = builder
        closure = getattr(function, "__closure__", None) or ()
        freevars = getattr(getattr(function, "__code__", None), "co_freevars", ())
        self._closure_names: dict[int, str] = {}
        for name, cell in zip(freevars, closure, strict=False):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            self._closure_names.setdefault(id(value), name)
        globals_map = getattr(function, "__globals__", {})
        referenced_names = getattr(getattr(function, "__code__", None), "co_names", ())
        self._global_names = {
            id(globals_map[name]): name for name in referenced_names if name in globals_map
        }
        self._constant_ids: dict[int, int] = {}
        # Retaining the objects makes identity deduplication sound: temporary
        # providers cannot recycle an id while this builder is alive.
        self._constant_values: dict[int, Any] = {}
        self.weak_constant_ids: set[int] = set()
        self.constants: list[ConstantRecord] = []

    def add_constant(self, value: Any, spec: ArraySpec) -> int:
        value_identity = id(value)
        existing = self._constant_ids.get(value_identity)
        if existing is not None and self._constant_values[value_identity] is value:
            if spec.weak:
                self.weak_constant_ids.add(existing)
            return existing

        dtype = normalize_constant_dtype(_dtype_name(spec.dtype))
        stored_value = (
            value
            if isinstance(value, _PortableConstant)
            else snapshot_constant_parts(value, shape=spec.shape, dtype=dtype)
        )
        if stored_value.shape != spec.shape or stored_value.dtype != dtype:
            raise ValueError("Staged constant parts do not match their abstract specification")
        node_id, native_digest = self._builder.append_constant(
            stored_value.data,
            spec.shape,
            dtype,
            kind=stored_value.kind,
        )

        byte_count = len(stored_value.data)
        if value_identity in self._closure_names:
            origin = "closure"
            name = self._closure_names[value_identity]
        elif value_identity in self._global_names:
            origin = "global"
            name = self._global_names[value_identity]
        else:
            origin = "created"
            name = None
        self.constants.append(
            ConstantRecord(
                value_id=node_id,
                origin=origin,
                location=_capture_location(),
                shape=spec.shape,
                dtype=dtype,
                bytes=byte_count,
                digest=native_digest,
                name=name,
            )
        )
        self._constant_ids[value_identity] = node_id
        self._constant_values[value_identity] = value
        if spec.weak:
            self.weak_constant_ids.add(node_id)
        return node_id


def _function_captures(function: Callable[..., Any]) -> Iterator[tuple[str, object]]:
    owner = getattr(function, "__self__", None)
    if owner is not None:
        yield "bound callable", owner
    closure = getattr(function, "__closure__", None) or ()
    code = getattr(function, "__code__", None)
    for name, cell in zip(getattr(code, "co_freevars", ()), closure, strict=False):
        try:
            yield name, cell.cell_contents
        except ValueError:
            continue
    globals_map = getattr(function, "__globals__", {})
    for name in getattr(code, "co_names", ()):
        if name in globals_map:
            yield name, globals_map[name]
    for index, value in enumerate(getattr(function, "__defaults__", None) or ()):
        yield f"default argument {index}", value
    for name, value in (getattr(function, "__kwdefaults__", None) or {}).items():
        yield f"default argument {name}", value


def _scalar_output_mask(
    graph: GraphStore,
    weak_source_ids: set[int],
) -> tuple[bool, ...]:
    """Propagate weak-scalar category through rank-zero numerical operations."""
    if not weak_source_ids:
        return tuple(False for _node_id in graph.outputs)

    weak = set(weak_source_ids)
    for node_id in graph.node_ids():
        if node_id in weak:
            continue
        node = graph.get_node(node_id)
        if not node.inputs or node.shape:
            continue
        if all(parent in weak for parent in node.inputs):
            weak.add(node_id)
    return tuple(node_id in weak for node_id in graph.outputs)


def _with_scalar_output_mask(
    artifact: _CompiledStage,
    *,
    offset: int,
    mask: Sequence[bool],
) -> _CompiledStage:
    """Override one contiguous semantic scalar-output region."""
    stop = offset + len(mask)
    if offset < 0 or stop > len(artifact.output_specs):
        msg = (
            "Staged scalar-output mask does not fit the transformed output: "
            f"offset={offset}, entries={len(mask)}, outputs={len(artifact.output_specs)}"
        )
        raise RuntimeError(msg)
    output_specs = list(artifact.output_specs)
    for index, weak in enumerate(mask, start=offset):
        spec = output_specs[index]
        if weak and spec.shape != ():
            msg = (
                "Only rank-zero staged outputs can restore Python scalar semantics; "
                f"output {index} has shape {spec.shape}"
            )
            raise RuntimeError(msg)
        output_specs[index] = replace(spec, weak=weak)
    return replace(artifact, output_specs=tuple(output_specs))


def _compile_stage(
    function: Callable[..., Any],
    call_tree: tuple[tuple[ArraySpec | StaticSpec, ...], dict[str, ArraySpec | StaticSpec]],
    *,
    array_api_version: str,
) -> _CompiledStage:
    graph_builder = create_graph_builder(required_array_api_version=array_api_version)
    stage_builder = _StageBuilder(function, graph_builder)
    array_factory = cast(
        "type[AbstractArray]",
        get_hook("advect.abstract_array_factory") or AbstractArray,
    )
    trace = AbstractTrace(
        graph_builder,
        profile=_ADVECT_ARRAY_SEMANTIC_PROFILE,
        array_api_version=array_api_version,
        add_constant=stage_builder.add_constant,
        array_factory=array_factory,
    )
    spec_leaves, call_treedef = tree_flatten(call_tree)
    stage_scope = array_factory._advect_stage_context(tuple(_function_captures(function)))
    call_specs: list[ArraySpec | StaticSpec] = []
    traced_leaves: list[Any] = []
    weak_input_ids: set[int] = set()
    for index, spec in enumerate(spec_leaves):
        if isinstance(spec, StaticSpec):
            call_specs.append(spec)
            traced_leaves.append(_snapshot_static_value(spec.value))
            continue
        if not isinstance(spec, ArraySpec):
            msg = (
                "stage specs must contain ArraySpec or StaticSpec leaves, "
                f"got {type(spec).__name__}"
            )
            raise TypeError(msg)
        # The artifact stores canonical dtype names; the trace sees the declared spec.
        stored_spec = replace(spec, dtype=_dtype_name(spec.dtype))
        call_specs.append(stored_spec)
        node_id = graph_builder.append_input_node(
            stored_spec.shape,
            stored_spec.dtype,
            name=f"arg{index}",
        )
        if spec.weak:
            weak_input_ids.add(node_id)
        traced_leaves.append(_new_abstract_array(trace, node_id, spec, owned=False))

    traced_args, traced_kwargs = tree_unflatten(call_treedef, traced_leaves)
    _set_active_recorder(
        graph_builder,
        trace_kind="stage_abstract",
        array_api_version=array_api_version,
    )
    try:
        with stage_scope:
            result = function(*traced_args, **traced_kwargs)
        output_leaves, output_treedef = tree_flatten(result)
        traced_output_leaves = [_lift(trace, leaf) for leaf in output_leaves]
        for leaf in traced_output_leaves:
            graph_builder.append_output(leaf.node_id)
        output_specs = tuple(leaf.spec for leaf in traced_output_leaves)
    finally:
        trace.open = False
        _set_active_recorder(None)

    graph, old_to_new, raw_optimization, raw_trace = graph_builder.finish()
    trace = StagedTrace(
        nodes=tuple(
            TracedNode(id=node_id, op=op, inputs=tuple(inputs), name=name)
            for node_id, op, inputs, name in raw_trace
        ),
        old_to_new=tuple(old_to_new),
        constants=tuple(stage_builder.constants),
    )
    constants: list[ConstantRecord] = []
    for record in stage_builder.constants:
        try:
            remapped_id = old_to_new[record.value_id]
        except IndexError as error:
            raise RuntimeError("Staged optimizer returned an incomplete ID remap") from error
        if remapped_id is None:
            continue
        constants.append(
            ConstantRecord(
                value_id=remapped_id,
                origin=record.origin,
                location=record.location,
                shape=record.shape,
                dtype=record.dtype,
                bytes=record.bytes,
                digest=record.digest,
                name=record.name,
            )
        )
    optimization = _decode_optimization(raw_optimization)
    weak_source_ids: set[int] = set()
    for raw_id in weak_input_ids | stage_builder.weak_constant_ids:
        try:
            remapped_id = old_to_new[raw_id]
        except IndexError as error:
            raise RuntimeError("Staged optimizer returned an incomplete ID remap") from error
        if remapped_id is not None:
            weak_source_ids.add(remapped_id)
    scalar_output_mask = _scalar_output_mask(graph, weak_source_ids)
    output_specs = tuple(
        replace(spec, dtype=_dtype_name(spec.dtype), weak=restore and spec.shape == ())
        for spec, restore in zip(output_specs, scalar_output_mask, strict=True)
    )

    return _CompiledStage(
        graph=graph,
        execution_plan=_bind_staged_execution(graph),
        call_treedef=call_treedef,
        call_specs=tuple(call_specs),
        output_treedef=output_treedef,
        output_specs=output_specs,
        constants=tuple(constants),
        optimization=optimization,
        trace=trace,
    )


def _validate_constant_manifest(
    graph: GraphStore,
    constants: Sequence[ConstantRecord],
) -> None:
    records: dict[int, ConstantRecord] = {}
    for record in constants:
        if records.setdefault(record.value_id, record) is not record:
            raise ValueError(f"Staged constant manifest repeats value %{record.value_id}")
    graph_ids = set(graph.constant_ids())
    if set(records) != graph_ids:
        missing = sorted(graph_ids - set(records))
        extra = sorted(set(records) - graph_ids)
        raise ValueError(
            f"Staged constant manifest does not match graph constants; "
            f"missing={missing}, extra={extra}"
        )
    for value_id, record in records.items():
        node = graph.get_node(value_id)
        # advect-runtime has already checked each payload against its node.
        if tuple(node.shape) != record.shape or _dtype_name(node.dtype) != _dtype_name(
            record.dtype
        ):
            raise ValueError(
                f"Staged constant %{value_id} manifest shape/dtype does not match its graph node"
            )
        _kind, _dtype, _shape, data, digest = graph._constant_parts(value_id)
        if len(data) != record.bytes:
            raise ValueError(
                f"Staged constant %{value_id} manifest metadata does not match its payload"
            )
        if digest != record.digest:
            raise ValueError(
                f"Staged constant %{value_id} manifest digest does not match its payload"
            )


def _deserialize_staged_graph(payload: object) -> GraphStore:
    """Link one already-optimized graph to the registry and load it without recompiling."""
    if not isinstance(payload, dict):
        raise TypeError("Staged graph payload must be a mapping")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TypeError("Staged graph nodes must be a list")
    registry = get_registry()
    # advect-runtime requires dense node ids, so a list position is its node id.
    custom_ids: list[int] = []
    for node_id, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise TypeError("Staged graph node must be a mapping")
        op = raw_node.get("op")
        schema_version = raw_node.get("schema_version")
        num_outputs = raw_node.get("num_outputs")
        if not isinstance(op, str):
            raise TypeError("Staged graph node op must be a string")
        if type(schema_version) is not int or schema_version < 1:
            raise TypeError("Staged graph node schema_version must be a positive integer")
        if type(num_outputs) is not int or num_outputs < 1:
            raise TypeError("Staged graph node num_outputs must be a positive integer")
        if op.startswith("custom."):
            try:
                _record_primitive_output_count(op, num_outputs)
            except KeyError as error:
                name = op.removeprefix("custom.")
                raise ValueError(f"Staged program requires unlinked primitive '{name}'") from error
            custom_ids.append(node_id)
        op_def = registry.get_optional(op)
        if op_def is None:
            raise ValueError(
                f"Op '{op}' is not registered. Import the required frontend or "
                "register the primitive before loading."
            )
        expected_schema = op_def.schema_version
        if schema_version != expected_schema:
            raise ValueError(
                f"Staged graph op '{op}' requires schema {schema_version}; "
                f"linked schema is {expected_schema}"
            )
        if op_def.num_outputs != num_outputs:
            raise ValueError(
                f"Op '{op}' expects num_outputs={op_def.num_outputs}, got num_outputs={num_outputs}"
            )

    graph = deserialize_graph_json(json.dumps(payload, separators=(",", ":")))
    _validate_custom_calls(graph, custom_ids)
    return graph


def _validate_custom_calls(graph: GraphStore, node_ids: Sequence[int]) -> None:
    """Validate the call metadata needed to link custom nodes safely."""
    for node_id in node_ids:
        node = graph.get_node(node_id)
        try:
            call_meta, _node_attrs = _split_primitive_attrs(
                decode_graph_attrs_from_native(node.attrs)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Staged custom node {node.op!r} has an invalid call contract"
            ) from error
        if call_meta.output_treedef.num_leaves != node.num_outputs:
            raise ValueError(
                f"Staged custom node {node.op!r} output structure does not match its arity"
            )
        if len(node.inputs) != sum(call_meta.input_leaf_mask):
            raise ValueError(
                f"Staged custom node {node.op!r} input count does not match its call structure"
            )


def _encode_artifact(artifact: _CompiledStage) -> dict[str, object]:
    return {
        "graph": json.loads(artifact.graph._to_json()),
        "call_treedef": _encode_treedef(artifact.call_treedef),
        "call_specs": [_encode_spec(spec) for spec in artifact.call_specs],
        "output_treedef": _encode_treedef(artifact.output_treedef),
        "output_specs": [_encode_spec(spec) for spec in artifact.output_specs],
        "constants": [_encode_constant(record) for record in artifact.constants],
        "optimization": _encode_optimization(artifact.optimization),
    }


def _decode_artifact(payload: object) -> _CompiledStage:
    if not isinstance(payload, dict):
        raise TypeError("Staged artifact must be a mapping")
    required = {
        "graph",
        "call_treedef",
        "call_specs",
        "output_treedef",
        "output_specs",
        "constants",
        "optimization",
    }
    if set(payload) != required:
        raise ValueError("Staged artifact has invalid fields")
    registry = get_registry()
    with registry.transaction():
        call_specs_payload = payload["call_specs"]
        output_specs_payload = payload["output_specs"]
        constants_payload = payload["constants"]
        if not isinstance(call_specs_payload, list):
            raise TypeError("Staged artifact call_specs must be a list")
        if not isinstance(output_specs_payload, list):
            raise TypeError("Staged artifact output_specs must be a list")
        if not isinstance(constants_payload, list):
            raise TypeError("Staged artifact constants must be a list")
        call_treedef = _decode_treedef(payload["call_treedef"])
        output_treedef = _decode_treedef(payload["output_treedef"])
        call_specs = tuple(_decode_spec(item) for item in call_specs_payload)
        decoded_output_specs = tuple(_decode_spec(item) for item in output_specs_payload)
        if any(not isinstance(spec, ArraySpec) for spec in decoded_output_specs):
            raise TypeError("Staged output specs must all be array specs")
        output_specs = cast("tuple[ArraySpec, ...]", decoded_output_specs)
        constants = tuple(_decode_constant(item) for item in constants_payload)
        optimization = _decode_optimization(payload["optimization"])
        if len(call_specs) != call_treedef.num_leaves:
            raise ValueError("Staged call specs do not match their pytree")
        if len(output_specs) != output_treedef.num_leaves:
            raise ValueError("Staged output specs do not match their pytree")
        graph = _deserialize_staged_graph(payload["graph"])
        input_specs = tuple(spec for spec in call_specs if isinstance(spec, ArraySpec))
        input_nodes = tuple(graph.get_node(node_id) for node_id in graph.inputs)
        if len(input_nodes) != len(input_specs) or any(
            tuple(node.shape) != spec.shape or _dtype_name(node.dtype) != spec.dtype
            for node, spec in zip(input_nodes, input_specs, strict=True)
        ):
            raise ValueError("Staged graph inputs do not match its call specs")
        if len(graph.outputs) != output_treedef.num_leaves:
            raise ValueError("Staged graph output count does not match its output pytree")
        for node_id, spec in zip(graph.outputs, output_specs, strict=True):
            node = graph.get_node(node_id)
            if tuple(node.shape) != spec.shape or _dtype_name(node.dtype) != spec.dtype:
                raise ValueError("Staged output specs do not match graph outputs")
        if graph.node_count != optimization.nodes_after:
            raise ValueError("Staged graph node count does not match its optimization report")
        _validate_constant_manifest(graph, constants)
        return _CompiledStage(
            graph=graph,
            execution_plan=_bind_staged_execution(graph),
            call_treedef=call_treedef,
            call_specs=call_specs,
            output_treedef=output_treedef,
            output_specs=output_specs,
            constants=constants,
            optimization=optimization,
        )


def _runtime_namespace(values: Sequence[Any], *, array_api_version: str) -> Any:
    """Select the provider for one call and require the program's Array API revision."""
    resolution = _negotiate_array_namespace_for_call(
        args=tuple(values),
        kwargs={},
        required_version=array_api_version,
    )
    if resolution is not None:
        namespace = resolution.raw_namespace
    else:
        resolve = get_hook("advect.default_array_namespace")
        if resolve is None:
            raise RuntimeError(
                "A staged call without provider-backed inputs requires a registered "
                "default array namespace"
            )
        namespace = resolve()
    if _get_backend_key_from_namespace(namespace) is None:
        raise TypeError("Staged array providers must expose a stable namespace name")
    version = _get_provider_array_api_version(namespace)
    reported_key = None if version is None else _version_key(version)
    requested_key = _version_key(array_api_version)
    if reported_key is None or requested_key is None or reported_key < requested_key:
        raise TypeError(
            f"Staged profile {_ADVECT_ARRAY_SEMANTIC_PROFILE!r} requires Array API "
            f"{array_api_version}; "
            f"the runtime provider exposes {version!r}"
        )
    return namespace


def _runtime_device(values: Sequence[Any]) -> tuple[object | None, str | None]:
    """Return the one device of the negotiated provider's inputs, if any."""
    selected: object | None = None
    selected_key: str | None = None
    for value in values:
        device = getattr(value, "device", None)
        if device is None:
            continue
        device_key = str(device)
        if selected_key is None:
            selected = device
            selected_key = device_key
        elif device_key != selected_key:
            raise TypeError(
                "A staged call cannot materialize constants across multiple devices; "
                f"got {selected_key!r} and {device_key!r}"
            )
    return selected, selected_key


def _coerce_constant(
    value: _PortableConstant,
    namespace: Any | None,
    *,
    device: object | None,
) -> Any:
    """Materialize one staged constant on a concrete or abstract provider."""
    dtype_name = value.dtype
    shape = value.shape
    if value.kind == "scalar":
        return next(iter_constant_values(value))
    if namespace is None:
        raise RuntimeError("Cannot materialize a staged array constant without an array namespace")
    abstract_materialize = getattr(namespace, "_advect_materialize_constant", None)
    if callable(abstract_materialize):
        return abstract_materialize(value, ArraySpec(shape, dtype_name))
    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        raise TypeError("The runtime array namespace does not provide asarray()")
    dtype = getattr(namespace, dtype_name, None)
    if dtype is None:
        raise TypeError(f"The runtime array namespace does not provide dtype {dtype_name!r}")
    kwargs: dict[str, object] = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    frombuffer = getattr(namespace, "frombuffer", None)
    materialized: Any
    if callable(frombuffer):
        # Some providers expose writable tensors over the supplied buffer and
        # warn (or reject) when handed immutable ``bytes``. Constants own this
        # invocation-local backing store, so a mutable copy is the portable
        # boundary.
        materialized = cast("Any", frombuffer(bytearray(value.data), dtype=dtype))
        if device is not None and str(getattr(materialized, "device", None)) != str(device):
            materialized = cast("Any", asarray(materialized, **kwargs))
    else:
        materialized = cast(
            "Any",
            asarray(tuple(iter_constant_values(value)), **kwargs),
        )
    if tuple(int(size) for size in materialized.shape) != shape:
        reshape = getattr(namespace, "reshape", None)
        if not callable(reshape):
            raise TypeError("The runtime array namespace does not provide reshape()")
        materialized = reshape(materialized, shape)
    actual_shape = tuple(int(size) for size in materialized.shape)
    actual_dtype = _dtype_name(materialized.dtype)
    if actual_shape != shape or actual_dtype != dtype_name:
        raise TypeError("The runtime provider did not preserve a staged constant's shape and dtype")
    return materialized


def _materialize_constants(
    compiled: _CompiledStage,
    state: _ExecutionState,
    namespace: Any,
    *,
    device: object | None,
    device_key: str | None,
) -> tuple[object, ...]:
    if not compiled.constants:
        return ()
    # A dynamic tracing namespace wraps the provider that owns the values.
    provider = getattr(namespace, "raw_namespace", namespace)

    def provider_values() -> tuple[object, ...]:
        return tuple(
            _coerce_constant(
                portable_constant_from_native(compiled.graph._constant_parts(node_id)),
                provider,
                device=device,
            )
            for node_id in compiled.graph.constant_ids()
        )

    if callable(getattr(provider, "_advect_materialize_constant", None)):
        # An abstract provider records the constants into its own staging trace.
        values = provider_values()
    else:
        with state.materialization_lock:
            cached_values = next(
                (
                    cached.values
                    for cached in state.materialized_constants
                    if cached.namespace is provider and cached.device == device_key
                ),
                None,
            )
            if cached_values is None:
                cached_values = provider_values()
                state.materialized_constants.append(
                    _MaterializedConstants(
                        namespace=provider,
                        device=device_key,
                        values=cached_values,
                    )
                )
            values = cached_values
    if provider is namespace:
        return values
    # Each dynamic trace lifts the provider arrays onto its own tape; Python
    # scalar constants need no provider identity.
    lift = namespace._advect_materialize_constant
    return tuple(lift(value, None) if hasattr(value, "shape") else value for value in values)


_STAGED_ERROR_NODE = re.compile(r"node %(\d+)")
_GRAPH_RENDER_LIMIT = 40


def _graph_node_text(graph: GraphStore, node_id: int, *, failed: bool = False) -> str:
    node = graph.get_node(node_id)
    marker = "->" if failed else "  "
    name = f" {node.name}" if node.name else ""
    inputs = ", ".join(f"%{parent}" for parent in node.inputs)
    return (
        f"{marker} %{node.id}{name} = {node.op}({inputs}) "
        f"shape={tuple(node.shape)}, dtype={node.dtype}"
    )


def _format_graph(graph: GraphStore) -> str:
    node_ids = tuple(graph.node_ids())
    if len(node_ids) > _GRAPH_RENDER_LIMIT:
        omitted = len(node_ids) - _GRAPH_RENDER_LIMIT
        node_ids = (*node_ids[:32], None, *node_ids[-8:])
    else:
        omitted = 0
    lines = [repr(graph)]
    lines.extend(
        (
            f"   ... {omitted} nodes omitted ..."
            if node_id is None
            else _graph_node_text(graph, node_id)
        )
        for node_id in node_ids
    )
    return "\n".join(lines)


def _staged_error_node_id(error: BaseException) -> int | None:
    notes = getattr(error, "__notes__", ())
    for note in reversed(notes):
        match = _STAGED_ERROR_NODE.search(str(note))
        if match is not None:
            return int(match.group(1))
    return None


def _add_staged_error_context(error: BaseException, graph: GraphStore) -> None:
    node_id = _staged_error_node_id(error)
    if node_id is None:
        return
    try:
        node = graph.get_node(node_id)
        parent_ids = tuple(node.inputs[-3:])
        lines = ["Advect graph context:"]
        lines.extend(_graph_node_text(graph, parent_id) for parent_id in parent_ids)
        lines.append(_graph_node_text(graph, node_id, failed=True))
        if node.source_location:
            lines.append(f"   source: {node.source_location}")
        note = "\n".join(lines)
    except Exception:  # noqa: BLE001 - preserve the original execution error
        return
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def _bind_staged_execution(graph: GraphStore) -> GraphExecutionPlan:
    try:
        return build_graph_execution_plan(graph, bind_native_node_evaluator)
    except Exception as error:
        _add_staged_error_context(error, graph)
        raise


def _execute_staged(
    compiled: _CompiledStage,
    state: _ExecutionState,
    inputs: Sequence[Any],
) -> list[Any]:
    array_api_version = compiled.graph.required_array_api_version
    namespace = _runtime_namespace(inputs, array_api_version=array_api_version)
    device, device_key = _runtime_device(inputs) if compiled.constants else (None, None)
    constants = _materialize_constants(
        compiled,
        state,
        namespace,
        device=device,
        device_key=device_key,
    )
    try:
        return execute_graph(
            compiled.execution_plan,
            inputs,
            constants,
            ResolvedArrayNamespace(namespace, array_api_version),
        )
    except Exception as error:
        _add_staged_error_context(error, compiled.graph)
        raise


class StagedProgram:
    """One callable input signature compiled into one immutable durable graph.

    Use ``stage`` to create a program. Its dictionary representation does
    not contain Python code and can be loaded after required primitives link.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> program = ad.stage(lambda x: x + 1, np.array([1.0, 2.0]))
    >>> loaded = ad.StagedProgram.from_dict(program.to_dict())
    >>> loaded(np.array([3.0, 4.0])).tolist()
    [4.0, 5.0]
    """

    __slots__ = ("_artifact", "_compile_seconds", "_execution_state")

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        specs: tuple[Any, ...],
        kw_specs: dict[str, Any],
        array_api_version: str,
    ) -> None:
        self._artifact, self._compile_seconds = self._compile(
            function,
            (specs, kw_specs),
            array_api_version=array_api_version,
        )
        self._execution_state = _ExecutionState()

    def __repr__(self) -> str:
        """Return a compact program summary for notebooks and debuggers."""
        return f"StagedProgram({self._artifact.graph!r})"

    def __str__(self) -> str:
        """Render the program's optimized operation sequence."""
        return _format_graph(self._artifact.graph)

    @staticmethod
    def _compile(
        function: Callable[..., Any],
        call_tree: tuple[
            tuple[Any, ...],
            dict[str, Any],
        ],
        *,
        array_api_version: str,
    ) -> tuple[_CompiledStage, float]:
        leaves, treedef = tree_flatten(call_tree)
        # The codec is the closed durability boundary for Static aux data and
        # mapping keys. Round-tripping here both validates and snapshots them.
        treedef = _decode_treedef(_encode_treedef(treedef))
        normalized_leaves = [
            StaticSpec(_snapshot_static_value(spec.value)) if isinstance(spec, StaticSpec) else spec
            for spec in leaves
        ]
        normalized_call_tree = cast(
            "tuple[tuple[ArraySpec | StaticSpec, ...], dict[str, ArraySpec | StaticSpec]]",
            tree_unflatten(treedef, normalized_leaves),
        )
        start = time.perf_counter()
        artifact = _compile_stage(
            function,
            normalized_call_tree,
            array_api_version=array_api_version,
        )
        return artifact, time.perf_counter() - start

    @property
    def graph(self) -> GraphStore:
        """Return this program's immutable graph."""
        return self._artifact.graph

    @property
    def signature(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return a detached positional and keyword input-specification tree."""
        treedef = _decode_treedef(_encode_treedef(self._artifact.call_treedef))
        specs = [_decode_spec(_encode_spec(spec)) for spec in self._artifact.call_specs]
        return cast(
            "tuple[tuple[Any, ...], dict[str, Any]]",
            tree_unflatten(treedef, specs),
        )

    @property
    def compile_seconds(self) -> float:
        """Return the time spent compiling this in-process program."""
        return self._compile_seconds

    @property
    def constants(self) -> tuple[ConstantRecord, ...]:
        """Return the concrete values captured by this program."""
        return self._artifact.constants

    @property
    def optimization(self) -> OptimizationReport:
        """Return this program's fixed-pipeline optimization report."""
        return self._artifact.optimization

    @property
    def trace(self) -> StagedTrace | None:
        """Return the staging tape and optimizer mapping for in-process programs.

        ``None`` for programs loaded from a durable artifact: the trace is a
        staging byproduct and is deliberately not serialized.
        """
        return self._artifact.trace

    @property
    def array_api_version(self) -> str:
        """Return the Array API revision required by this program."""
        return self._artifact.graph.required_array_api_version

    def to_dict(self) -> dict[str, object]:
        """Serialize this program without serializing Python code."""
        return {
            "format": _STAGED_PROGRAM_FORMAT,
            "version": _STAGED_PROGRAM_FORMAT_VERSION,
            "program": _encode_artifact(self._artifact),
        }

    @classmethod
    def from_dict(cls, payload: object) -> StagedProgram:
        """Load a versioned staged artifact after linking custom primitives."""
        if not isinstance(payload, dict):
            raise TypeError("Staged program payload must be a mapping")
        if set(payload) != {"format", "version", "program"}:
            raise ValueError("Staged program payload has invalid fields")
        if payload["format"] != _STAGED_PROGRAM_FORMAT:
            raise ValueError(f"Unknown staged program format {payload['format']!r}")
        format_version = payload["version"]
        if type(format_version) is not int:
            raise TypeError("Staged program format version must be an integer")
        if format_version != _STAGED_PROGRAM_FORMAT_VERSION:
            raise ValueError(f"Unsupported staged program format version {format_version}")
        artifact = _decode_artifact(payload["program"])
        loaded = cls.__new__(cls)
        loaded._artifact = artifact
        loaded._compile_seconds = 0.0
        loaded._execution_state = _ExecutionState()
        return loaded

    def _staged_transform(
        self,
        function: Callable[..., Any],
        *,
        output_argname: str | None = None,
        scalar_output_override: tuple[int, tuple[bool, ...]] | None = None,
    ) -> StagedProgram:
        """Compile a graph-producing transform for this program's signature."""
        artifact = self._artifact
        call_tree = cast(
            "tuple[tuple[ArraySpec | StaticSpec, ...], dict[str, ArraySpec | StaticSpec]]",
            tree_unflatten(artifact.call_treedef, list(artifact.call_specs)),
        )
        if output_argname is not None:
            args, kwargs = call_tree
            if output_argname in kwargs:
                raise ValueError(f"Staged transform reserves keyword argument {output_argname!r}")
            output_tree = tree_unflatten(
                artifact.output_treedef,
                list(artifact.output_specs),
            )
            call_tree = (args, {**kwargs, output_argname: output_tree})
        transformed = self.__class__.__new__(self.__class__)
        transformed._artifact, transformed._compile_seconds = self._compile(
            function,
            call_tree,
            array_api_version=artifact.graph.required_array_api_version,
        )
        if scalar_output_override is not None:
            offset, mask = scalar_output_override
            transformed._artifact = _with_scalar_output_mask(
                transformed._artifact,
                offset=offset,
                mask=mask,
            )
        transformed._execution_state = _ExecutionState()
        return transformed

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        artifact = self._artifact
        concrete_leaves = _flatten_runtime_to_treedef((args, kwargs), artifact.call_treedef)
        runtime_inputs: list[Any] = []
        restore_scalars = False
        for index, (spec, value) in enumerate(
            zip(artifact.call_specs, concrete_leaves, strict=True)
        ):
            if isinstance(spec, StaticSpec):
                if not _same_static_value(value, spec.value):
                    raise TypeError(f"Static staged argument leaf {index} changed value")
                continue
            if spec.weak and type(value) in {bool, complex, float, int}:
                runtime_inputs.append(_normalize_weak_runtime_scalar(value, spec))
                restore_scalars = True
                continue
            actual = _value_spec(value)
            if (
                actual.shape != spec.shape
                or actual.dtype != spec.dtype
                or actual.weak != spec.weak
                or (spec.device is not None and actual.device != spec.device)
            ):
                raise ValueError(
                    f"Staged argument leaf {index} expected shape={spec.shape}, "
                    f"dtype={spec.dtype}, device={spec.device}, weak={spec.weak}; "
                    f"got shape={actual.shape}, dtype={actual.dtype}, "
                    f"device={actual.device}, weak={actual.weak}"
                )
            runtime_inputs.append(value)

        outputs = _execute_staged(artifact, self._execution_state, runtime_inputs)
        for index, spec in enumerate(artifact.output_specs):
            if not spec.weak:
                continue
            leaf = outputs[index]
            mark_weak = getattr(leaf, "_advect_mark_weak", None)
            if callable(mark_weak):
                mark_weak()
            item = getattr(leaf, "item", None)
            if restore_scalars and getattr(leaf, "shape", None) == () and callable(item):
                outputs[index] = item()
        return tree_unflatten(artifact.output_treedef, outputs)


def stage(
    function: Callable[..., Any] | None = None,
    *examples: Any,
    specs: tuple[Any, ...] | None = None,
    kw_specs: dict[str, Any] | None = None,
    array_api_version: str | None = None,
) -> StagedProgram | Callable[[Callable[..., Any]], StagedProgram]:
    """Compile one callable signature into an immutable staged program.

    Use the direct form with a callable, or omit ``function`` to create a
    decorator. Applying the decorator compiles the function and replaces it
    with a ``StagedProgram``. Compilation traces the Python callable once with
    abstract values; later calls execute the graph without running or
    retracing the Python callable.

    Declare the positional signature in exactly one of two ways: pass concrete
    ``examples`` to infer its array leaves, or pass ``specs`` containing
    ``ArraySpec`` and ``StaticSpec`` leaves. Keyword arguments have no example
    form and are declared with ``kw_specs`` in either case.

    Parameters
    ----------
    function
        Callable to compile. If omitted, return a decorator that compiles the
        callable it receives.
    *examples
        Concrete positional arguments whose pytree structure, shapes, dtypes,
        devices, and Python-scalar categories define the compiled signature.
        Wrap a non-array compile-time leaf in ``StaticSpec``. Mutually
        exclusive with ``specs``.
    specs
        Positional argument specification tree. Every leaf must be an
        ``ArraySpec`` or ``StaticSpec``. Mutually exclusive with ``examples``.
    kw_specs
        Mapping from keyword argument names to specification trees whose
        leaves are ``ArraySpec`` or ``StaticSpec``. The mapping is combined
        with the positional signature declared by ``examples`` or ``specs``.
    array_api_version
        Array API revision to compile and store in the graph. With concrete
        examples and no explicit revision, Advect selects the newest supported
        revision served by their common array provider. With ``specs`` alone,
        it selects Advect's latest supported revision. An explicit revision
        must be supported by Advect and by the provider of every array example.

    Returns
    -------
    StagedProgram or callable
        A fully compiled, single-signature ``StagedProgram`` when ``function``
        is supplied; otherwise, a decorator that returns such a program. The
        program snapshots static inputs and captured constants, can be
        serialized, and never grows a polymorphic cache or retraces.

    Raises
    ------
    TypeError
        If neither ``examples`` nor ``specs`` is supplied, if both are
        supplied, if an example is neither array-like nor a supported Python
        scalar nor wrapped in ``StaticSpec``, if a specification contains
        another leaf type, or if the concrete array examples cannot use one
        common provider at the selected Array API revision.
    ValueError
        If ``array_api_version`` is not a supported revision, or if abstract
        tracing finds incompatible shapes, dtypes, or operation semantics.

    Notes
    -----
    A returned program accepts only its compiled call pytree and leaf
    contract. At execution time, a changed call structure, non-array leaf, or
    static value raises ``TypeError``; an incompatible array shape, dtype,
    device, or Python-scalar category raises ``ValueError``.

    Examples
    --------
    Infer a direct-call signature from a concrete array:

    >>> import advect as ad
    >>> import numpy as np
    >>> def add_one(x):
    ...     return x + 1
    >>> example_input = np.array([1.0, 2.0])
    >>> program = ad.stage(add_one, example_input)
    >>> program(np.array([3.0, 4.0])).tolist()
    [4.0, 5.0]

    The decorator form uses explicit positional and keyword specifications:

    >>> @ad.stage(
    ...     specs=(ad.ArraySpec((2,), "float32"),),
    ...     kw_specs={"scale": ad.ArraySpec((), "float32")},
    ... )
    ... def scale(x, *, scale):
    ...     return x * scale
    >>> scale(
    ...     np.array([1.0, 2.0], dtype=np.float32),
    ...     scale=np.asarray(2.0, dtype=np.float32),
    ... ).tolist()
    [2.0, 4.0]
    """
    if examples:
        if specs is not None:
            raise TypeError("stage() accepts example arguments or specs=..., not both")
        positional_specs = _specs_from_examples(examples)
    elif specs is not None:
        positional_specs = specs
    else:
        raise TypeError(
            "stage() requires example arguments or specs=... so it can return "
            "one fully compiled single-signature StagedProgram"
        )

    if array_api_version is not None:
        materialize_array_api_profile(array_api_version)
    resolution = (
        _negotiate_array_namespace_for_call(
            args=examples,
            kwargs={},
            required_version=array_api_version,
        )
        if examples
        else None
    )
    selected_array_api_version = (
        resolution.requested_version
        if resolution is not None
        else array_api_version or LATEST_ARRAY_API_VERSION
    )

    def decorate(fn: Callable[..., Any]) -> StagedProgram:
        return StagedProgram(
            fn,
            specs=positional_specs,
            kw_specs={} if kw_specs is None else dict(kw_specs),
            array_api_version=selected_array_api_version,
        )

    return decorate if function is None else decorate(function)


def call_primitive_abstract(
    primitive: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Emit a unified custom primitive from an active abstract trace."""
    static_arguments = {name: kwargs[name] for name in primitive.static_argnames if name in kwargs}
    dynamic_kwargs = {name: value for name, value in kwargs.items() if name not in static_arguments}
    paths, call_leaves, call_treedef = tree_flatten_with_paths((args, dynamic_kwargs))
    direct_inputs = [value for value in call_leaves if isinstance(value, AbstractArray)]
    if not direct_inputs:
        return primitive._dispatch_impl(*args, **kwargs)
    trace = direct_inputs[0]._trace

    abstract_inputs: list[AbstractArray] = []
    abstract_call_leaves: list[Any] = []
    input_leaf_mask: list[bool] = []
    static_leaves: list[Any] = []
    nondiff_input_mask: list[bool] = []
    dynamic_argnames = frozenset(primitive._dynamic_argnames)
    nondiff_argnames = frozenset(primitive.nondiff_argnames)
    for path, value in zip(paths, call_leaves, strict=True):
        parameter = _keyword_parameter(path)
        if isinstance(value, AbstractArray):
            value._require()
            if value._trace is not trace:
                raise TypeError("Cannot mix primitive arguments from different abstract traces")
            abstract_value = value
        elif isinstance(value, (bool, int, float, complex)) or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            abstract_value = _lift(trace, value)
        else:
            if parameter in dynamic_argnames:
                msg = (
                    f"Primitive '{primitive.name}' argument '{parameter}' is not "
                    "traceable; declare it in static_argnames or pass an array/scalar"
                )
                raise TypeError(msg)
            input_leaf_mask.append(False)
            static_leaves.append(value)
            abstract_call_leaves.append(value)
            continue
        abstract_inputs.append(abstract_value)
        abstract_call_leaves.append(AbstractValue(abstract_value.spec))
        input_leaf_mask.append(True)
        nondiff_input_mask.append(parameter in nondiff_argnames)

    abstract_rule = primitive._abstract_rule
    if abstract_rule is None:
        from advect.core._primitive import MissingPrimitiveRuleError  # noqa: PLC0415

        msg = (
            f"Primitive '{primitive.name}' is missing 'abstract'; "
            "define it with @primitive.def_abstract."
        )
        raise MissingPrimitiveRuleError(msg)
    abstract_args, abstract_kwargs = tree_unflatten(call_treedef, abstract_call_leaves)
    abstract_kwargs.update(static_arguments)
    result = abstract_rule(*abstract_args, **abstract_kwargs)
    result_specs, output_treedef = tree_flatten(result)
    if not result_specs:
        raise TypeError(f"Primitive '{primitive.name}' abstract rule returned no values")
    normalized: list[ArraySpec] = []
    for item in result_specs:
        spec = item.spec if isinstance(item, AbstractValue) else item
        if not isinstance(spec, ArraySpec):
            msg = f"Primitive '{primitive.name}' abstract rule must return ArraySpec leaves"
            raise TypeError(msg)
        normalized.append(spec)
    call_meta = _PrimitiveCallMeta(
        call_treedef=call_treedef,
        input_leaf_mask=tuple(input_leaf_mask),
        static_leaves=tuple(static_leaves),
        output_treedef=output_treedef,
        nondiff_input_mask=tuple(nondiff_input_mask),
    )
    attrs = {
        _PRIMITIVE_CALL_KEY: call_meta,
        **static_arguments,
    }
    _record_primitive_output_count(primitive.op_name, len(normalized))
    node_id = _append_node(
        trace,
        op=primitive.op_name,
        inputs=tuple(value.node_id for value in abstract_inputs),
        attrs=attrs,
        shape=normalized[0].shape,
        dtype=normalized[0].dtype,
        num_outputs=len(normalized),
        output_shapes=tuple(spec.shape for spec in normalized) if len(normalized) > 1 else None,
        output_dtypes=tuple(spec.dtype for spec in normalized) if len(normalized) > 1 else None,
    )
    if len(normalized) == 1:
        return tree_unflatten(
            output_treedef,
            [_new_abstract_array(trace, node_id, normalized[0])],
        )
    outputs: list[AbstractArray] = []
    for index, spec in enumerate(normalized):
        output_id = _append_node(
            trace,
            op="advect.getoutput",
            inputs=(node_id,),
            attrs={"index": index, "num_outputs": len(normalized)},
            shape=spec.shape,
            dtype=spec.dtype,
        )
        outputs.append(_new_abstract_array(trace, output_id, spec))
    return tree_unflatten(output_treedef, outputs)


__all__ = [
    "ArraySpec",
    "ConstantRecord",
    "OptimizationPass",
    "OptimizationReport",
    "StagedProgram",
    "StaticSpec",
    "stage",
]
