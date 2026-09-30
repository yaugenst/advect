"""Concrete define-by-run linearization on the lightweight SSA tape."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import batched
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self, cast

from advect.autodiff.api._jvp_seeds import (
    _build_input_tangent_seeds,
)
from advect.autodiff.api._pullback_values import (
    _build_grad_outputs,
    _build_grad_tree,
    _coerce_output_cotangent_like,
    _flatten_output_cotangents,
    _format_backward_result,
    _unbroadcast,
    _zeros_like,
)
from advect.autodiff.api._scalar_boundary import _unlift_scalar_tree_by_mask
from advect.autodiff.api.inputs import (
    _array_namespace_for_input,
    _normalize_argnums_for_call,
    _trace_selected_args_and_kwargs,
    _trace_value_as_inputs,
)
from advect.autodiff.api.trace import _mark_outputs
from advect.autodiff.rules.array_family._backend_runtime import (
    _maybe_unwrap_array_family_jvp_rule,
    run_with_array_family_backend_provider,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _dtype_is_complex,
    dtype_is_inexact,
)
from advect.autodiff.rules.array_family.jvp.common import _astype_preserving_trace
from advect.autodiff.rules.array_family.providers import (
    try_resolve_array_family_backend_provider,
)
from advect.core._abstract_helpers import PYTHON_SCALAR_TYPES, dtype_name
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.core._array_api.providers import (
    _PRIMITIVE_TYPES,
    _get_array_namespace,
    _has_instance_namespace_override,
    _negotiate_array_namespace_for_call,
    _negotiate_default_array_namespace,
    _type_level_cache,
)
from advect.core._array_protocol_helpers import WEAK_SCALAR_OPS
from advect.core._backends import dispatch_input
from advect.core._context import (
    _is_numerics_debug,
    _numerics_context,
    _set_active_recorder,
    _use_array_api_version,
)
from advect.core._diagnostics import check_tape_numerics, raise_if_nonfinite
from advect.core._errors import AdvectError, NoJVPError, NoVJPError
from advect.core._eval_dispatch import _decode_attrs_for_vjp
from advect.core._native import (
    DynamicTape,
    dynamic_jvp,
    dynamic_jvp_many,
    dynamic_vjp,
    dynamic_vjp_many,
)
from advect.core._pytree import _get_node_impl, tree_flatten, tree_unflatten
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider
    from advect.core._pytree import TreeDef
    from advect.core._registry_types import OpDef


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Concrete values and call structure retained by one linearization."""

    tape: DynamicTape
    positional_specs: list[Any]
    named_specs: dict[str, Any]
    output_treedef: Any
    output_ids: tuple[int, ...]
    output: Any
    provider: ArrayFamilyBackendProvider | None = None
    input_primals: tuple[Any, ...] = ()
    restore_scalar_outputs: tuple[bool, ...] = ()
    array_api_version: str = LATEST_ARRAY_API_VERSION


@dataclass(frozen=True, slots=True)
class _TransposeRule:
    definition: OpDef
    vjp: Callable[..., tuple[Any | None, ...]] | None
    selective_vjp: Callable[..., tuple[Any | None, ...]] | None
    reads_arrays: bool


_BATCH_VJP_ATTR = "__advect_vjp_many__"


def _rules_read_arrays(op: str, definition: OpDef) -> bool:
    """Return whether the built-in rules of *op* read Python scalars as arrays.

    The native tape presents a weak operand as a Python scalar, which the
    elementwise rules combine by NEP 50 promotion. Every other built-in rule
    reads its values' shapes and dtypes, as a NumPy function reads a Python
    scalar through a rank-zero array. A user primitive's rules receive the
    Python scalar itself.
    """
    rule = definition.jvp
    return (
        op not in WEAK_SCALAR_OPS
        and rule is not None
        and _maybe_unwrap_array_family_jvp_rule(rule) is not None
    )


def _as_rule_arrays(values: tuple[object, ...]) -> tuple[object, ...]:
    """Represent each Python scalar among *values* as a rank-zero provider array."""
    for value in values:
        if type(value) in PYTHON_SCALAR_TYPES:
            return tuple(
                xp.asarray(item) if type(item) in PYTHON_SCALAR_TYPES else item for item in values
            )
    return values


def _transpose_rule(op: str) -> _TransposeRule:
    definition = get_registry().get(op)
    vjp = definition.vjp
    selective = None if vjp is None else getattr(vjp, "__advect_vjp_for_input_indices__", None)
    return _TransposeRule(
        definition=definition,
        vjp=vjp,
        selective_vjp=(
            cast("Callable[..., tuple[Any | None, ...]]", selective)
            if callable(selective)
            else None
        ),
        reads_arrays=_rules_read_arrays(op, definition),
    )


class _DynamicBindingCache:
    __slots__ = ("binding_vectors", "jvp", "registry", "revision", "vjp")

    def __init__(self) -> None:
        self.registry: object | None = None
        self.revision = -1
        self.jvp: dict[str, Callable[..., object]] = {}
        self.vjp: dict[str, Callable[..., object]] = {}
        self.binding_vectors: dict[
            tuple[str, ...],
            tuple[
                tuple[Callable[..., object] | None, ...],
                tuple[Callable[..., object] | None, ...],
                tuple[tuple[bool, bool, bool] | None, ...],
            ],
        ] = {}


_DYNAMIC_BINDING_CACHE = _DynamicBindingCache()


def _dynamic_binding_cache() -> _DynamicBindingCache:
    registry = get_registry()
    revision = registry.get_revision()
    cache = _DYNAMIC_BINDING_CACHE
    if cache.registry is not registry or cache.revision != revision:
        cache.registry = registry
        cache.revision = revision
        cache.jvp = {}
        cache.vjp = {}
        cache.binding_vectors = {}
    return cache


def _checked_numerics[T](
    value: T,
    *,
    phase: str,
    op: str,
    source_location: str | None,
) -> T:
    if _is_numerics_debug():
        raise_if_nonfinite(value, phase=phase, op=op, source_location=source_location)
    return value


_RANK_ZERO_AXES = (0, -1, (0,), (-1,))


def _decode_rule_attrs(
    op: str,
    raw_attrs: object,
    operands: tuple[object, ...],
    result: object,
) -> dict[str, Any]:
    """Decode a node's attributes for its derivative rules.

    NumPy's reductions and ``squeeze`` accept the axis 0 or -1 of a rank-0
    array and reduce or remove nothing. A node that maps its one rank-0
    operand to a rank-0 result through such an axis is exactly that case, so
    its rules receive the empty axis tuple. A gather such as ``take`` carries
    its indices as a second operand and keeps its axis.
    """
    if raw_attrs is None:
        return {}
    attrs = _decode_attrs_for_vjp(op, cast("dict[str, Any]", raw_attrs))
    axis = attrs.get("axis")
    if (
        isinstance(axis, (int, tuple))
        and axis in _RANK_ZERO_AXES
        and len(operands) == 1
        and not getattr(operands[0], "shape", ())
        and not getattr(result, "shape", ())
    ):
        attrs["axis"] = ()
    return attrs


def _jvp_binding(op: str) -> Callable[..., object]:
    cache = _dynamic_binding_cache()
    cached = cache.jvp.get(op)
    if cached is not None:
        return cached
    definition = get_registry().get(op)
    rule = definition.jvp
    active_rule = None
    if rule is not None:
        active_rule = _maybe_unwrap_array_family_jvp_rule(rule) or rule
    reads_arrays = _rules_read_arrays(op, definition)

    def apply(
        answer: object,
        operands: tuple[object, ...],
        tangents: tuple[object | None, ...],
        raw_attrs: object,
        source_location: str | None,
    ) -> object:
        if active_rule is None:
            if definition.non_differentiable_reason is not None and _is_locally_constant(
                answer, operands, tangents
            ):
                return None
            msg = f"Cannot linearize primitive '{op}': no JVP rule is installed"
            raise NoJVPError(msg, op=op, source_location=source_location)
        attrs = _decode_rule_attrs(op, raw_attrs, operands, answer)
        if reads_arrays:
            operands = _as_rule_arrays(operands)
            tangents = _as_rule_arrays(tangents)
        with _numerics_context("JVP propagation", source_location):
            return _checked_numerics(
                _project_output_tangent(
                    answer,
                    active_rule(answer, *operands, tangents=tangents, **attrs),
                ),
                phase="JVP propagation",
                op=op,
                source_location=source_location,
            )

    cache.jvp[op] = apply
    return apply


def _vjp_binding(op: str) -> Callable[..., object]:
    cache = _dynamic_binding_cache()
    cached = cache.vjp.get(op)
    if cached is not None:
        return cached
    rule = _transpose_rule(op)

    def apply(
        answer: object,
        operands: tuple[object, ...],
        cotangent: object,
        raw_attrs: object,
        active_positions: tuple[int, ...],
        residual: object,
        parent_specs: tuple[tuple[tuple[int, ...], object] | None, ...],
        source_location: str | None,
    ) -> list[object | None]:
        with _numerics_context("VJP propagation", source_location):
            return _checked_numerics(
                _apply_vjp_binding(
                    op=op,
                    source_location=source_location,
                    rule=rule,
                    answer=answer,
                    operands=operands,
                    cotangent=cotangent,
                    raw_attrs=raw_attrs,
                    active_positions=active_positions,
                    residual=residual,
                    parent_specs=parent_specs,
                ),
                phase="VJP propagation",
                op=op,
                source_location=source_location,
            )

    if (
        rule.vjp is None
        and rule.definition.jvp is not None
        and rule.definition.non_differentiable_reason is None
    ):

        def apply_many(
            answer: object,
            operands: tuple[object, ...],
            cotangents: tuple[object, ...],
            raw_attrs: object,
            active_positions: tuple[int, ...],
            residual: object,
            parent_specs: tuple[tuple[tuple[int, ...], object] | None, ...],
            source_location: str | None,
        ) -> tuple[list[object | None], ...]:
            del residual
            with _numerics_context("VJP propagation", source_location):
                return _checked_numerics(
                    _apply_structural_vjp_binding_many(
                        op=op,
                        source_location=source_location,
                        rule=rule,
                        answer=answer,
                        operands=operands,
                        cotangents=cotangents,
                        raw_attrs=raw_attrs,
                        active_positions=active_positions,
                        parent_specs=parent_specs,
                    ),
                    phase="VJP propagation",
                    op=op,
                    source_location=source_location,
                )

        setattr(apply, _BATCH_VJP_ATTR, apply_many)

    cache.vjp[op] = apply
    return apply


def _freeze_dynamic_tape(tape: DynamicTape) -> None:
    binding_cache = _dynamic_binding_cache()
    op_names = tuple(tape.op_names)
    cached_bindings = binding_cache.binding_vectors.get(op_names)
    if cached_bindings is not None:
        tape.freeze(*cached_bindings)
        return

    jvp_bindings: list[Callable[..., object] | None] = []
    vjp_bindings: list[Callable[..., object] | None] = []
    reverse_needs: list[tuple[bool, bool, bool] | None] = []
    registry = get_registry()
    for op in op_names:
        internal = op in {"advect.input", "advect.const"}
        if internal:
            jvp_bindings.append(None)
            vjp_bindings.append(None)
            reverse_needs.append(None)
            continue
        definition = registry.get(op)
        jvp_bindings.append(_jvp_binding(op))
        vjp_bindings.append(_vjp_binding(op))
        explicit_vjp = definition.vjp is not None
        reverse_needs.append(
            (
                definition.vjp_needs_output if explicit_vjp else True,
                definition.vjp_needs_inputs if explicit_vjp else True,
                definition.has_residual,
            )
        )
    binding_vector = (tuple(jvp_bindings), tuple(vjp_bindings), tuple(reverse_needs))
    tape.freeze(*binding_vector)
    binding_cache.binding_vectors[op_names] = binding_vector


@contextmanager
def _dynamic_trace(
    *,
    array_api_version: str,
    reverse_only: bool = False,
    require_jvp: bool = False,
) -> Generator[DynamicTape, None, None]:
    tape = DynamicTape()
    _set_active_recorder(
        cast("Any", tape),
        trace_kind="autodiff_dynamic",
        array_api_version=array_api_version,
        require_jvp=require_jvp,
    )
    error: BaseException | None = None
    try:
        try:
            yield tape
        except BaseException as caught:  # noqa: BLE001 - finalize on every interpreter exit
            error = caught
        else:
            try:
                _freeze_dynamic_tape(tape)
            except BaseException as caught:  # noqa: BLE001 - preserve failed trace cleanup
                error = caught
        try:
            _set_active_recorder(None)
        except BaseException as finalization_error:
            with suppress(Exception):
                tape.release_payloads()
            if error is None:
                raise
            error.add_note(f"Dynamic trace finalization also failed: {finalization_error}")
        if error is None and reverse_only:
            try:
                tape.prune_reverse_payloads()
            except BaseException as caught:  # noqa: BLE001 - preserve cleanup below
                error = caught
        if error is not None:
            raise error.with_traceback(error.__traceback__)
    finally:
        if error is not None:
            with suppress(Exception):
                tape.release_payloads()


def trace_call(
    f: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argnums: tuple[int, ...],
    argnames: tuple[str, ...] | None,
    reverse_only: bool = False,
    require_jvp: bool = False,
    select_output: Callable[[Any], Any] | None = None,
) -> TraceResult:
    """Trace one concrete call without constructing a durable graph.

    ``select_output`` receives the traced result of ``f`` and returns the
    differentiated output, so a caller can split off an auxiliary sidecar
    while input selection still follows the signature of ``f`` itself.
    """

    def trace_selected_call(
        tape: DynamicTape,
        xp: object | None,
    ) -> tuple[object, list[Any], dict[str, Any]]:
        (
            traced_args,
            traced_kwargs,
            positional_specs,
            named_specs,
        ) = _trace_selected_args_and_kwargs(
            f,
            graph=tape,
            args=args,
            kwargs=kwargs,
            normalized_argnums=_normalize_argnums_for_call(argnums, nargs=len(args)),
            argnames=argnames,
            xp=xp,
        )
        traced_output = f(*traced_args, **traced_kwargs)
        if select_output is not None:
            traced_output = select_output(traced_output)
        return traced_output, positional_specs, named_specs

    return _trace(
        trace_selected_call,
        args=args,
        kwargs=kwargs,
        reverse_only=reverse_only,
        require_jvp=require_jvp,
    )


def _trace(
    run: Callable[[DynamicTape, object | None], tuple[object, list[Any], dict[str, Any]]],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reverse_only: bool = False,
    require_jvp: bool = False,
) -> TraceResult:
    """Record ``run`` on a fresh tape under the revision negotiated for ``args``.

    ``run`` receives the tape and the raw namespace that lifts Python scalars.
    It traces its inputs and returns the traced output together with the
    positional and named input specifications.
    """
    resolution = _negotiate_array_namespace_for_call(args=args, kwargs=kwargs)
    if resolution is None:
        resolution = _negotiate_default_array_namespace()
    selected_version = (
        LATEST_ARRAY_API_VERSION if resolution is None else resolution.requested_version
    )
    with _dynamic_trace(
        array_api_version=selected_version,
        reverse_only=reverse_only,
        require_jvp=require_jvp,
    ) as tape:
        traced_output, positional_specs, named_specs = run(
            tape,
            None if resolution is None else resolution.raw_namespace,
        )
        output_treedef, output_ids = _mark_outputs(traced_output, cast("Any", tape))
        output_values = tape.values(output_ids)
        if _is_numerics_debug():
            check_tape_numerics(tape)
        provider = _trace_provider(
            (*tape.values(tape.inputs), *output_values),
            array_api_version=selected_version,
        )

    return TraceResult(
        tape=tape,
        positional_specs=positional_specs,
        named_specs=named_specs,
        output_treedef=output_treedef,
        output_ids=tuple(output_ids),
        output=_concrete_output(tape, output_treedef, output_values),
        provider=provider,
        restore_scalar_outputs=tuple(tape.weak_mask(output_ids)),
        array_api_version=selected_version,
    )


def _concrete_output(tape: DynamicTape, treedef: TreeDef, values: list[Any]) -> Any:  # noqa: ANN401 - output pytree
    """Restore a closed trace's output pytree, releasing the tape on failure."""
    try:
        return tree_unflatten(treedef, values)
    except Exception:
        tape.release_payloads()
        raise


_PROVIDER_CACHE_MISS = object()
_TRACE_PROVIDERS: dict[tuple[tuple[type[Any], ...], str], ArrayFamilyBackendProvider | None] = (
    _type_level_cache({})
)


def _namespace_follows_type(value: object) -> bool:
    """Return whether core resolves a value's Array API namespace from its type alone.

    This is the rule behind core's per-type namespace cache: primitives have no
    namespace, and any other value's type must define `__array_namespace__`
    without an instance override. Namespaces found through instance attribute
    lookup or the fallback provider can differ between values of one type.
    """
    value_type = type(value)
    if value is None or value_type in _PRIMITIVE_TYPES:
        return True
    return not (
        getattr(value_type, "__array_namespace__", None) is None
        or _has_instance_namespace_override(value)
        or getattr(value_type, "__advect_namespace_is_instance_specific__", False)
    )


def _trace_provider(
    values: tuple[object, ...],
    *,
    array_api_version: str,
) -> ArrayFamilyBackendProvider | None:
    """Resolve one derivative provider for a closed trace's inputs and outputs.

    Concrete values whose namespace follows from their type share a result per
    distinct value types. Tracers and instance-specific namespaces resolve every
    time, and only module namespaces are retained, since trace-aware namespace
    proxies are invocation-local.
    """
    key = None
    if all(
        _namespace_follows_type(value) and not callable(getattr(value, "_advect_snapshot", None))
        for value in values
    ):
        key = (tuple(dict.fromkeys(type(value) for value in values)), array_api_version)
        cached = _TRACE_PROVIDERS.get(key, _PROVIDER_CACHE_MISS)
        if cached is not _PROVIDER_CACHE_MISS:
            return cast("ArrayFamilyBackendProvider | None", cached)
    try:
        provider = try_resolve_array_family_backend_provider(
            *values,
            array_api_version=array_api_version,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        provider = None
    if key is not None and (provider is None or isinstance(provider.namespace, ModuleType)):
        _TRACE_PROVIDERS[key] = provider
    return provider


def unary_array_trace_provider(
    value: object,
) -> tuple[bool, ArrayFamilyBackendProvider | None]:
    """Recognize one array leaf and resolve its derivative provider once.

    This recognizes the dominant dynamic-gradient case without flattening a
    pytree. Namespace resolution still enforces the pinned provider contract.
    """
    if _get_node_impl(type(value)) is not None:
        return False, None
    resolution = _negotiate_array_namespace_for_call(args=(value,), kwargs={})
    if (
        resolution is None
        or _array_namespace_for_input(
            value,
            array_api_version=resolution.requested_version,
        )
        is None
    ):
        return False, None
    return True, try_resolve_array_family_backend_provider(
        value,
        array_api_version=resolution.requested_version,
    )


def trace_unary_array_call(
    f: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_name: str,
    provider: ArrayFamilyBackendProvider | None,
) -> TraceResult:
    """Trace a call whose sole selected input is one positional array leaf."""
    resolution = _negotiate_array_namespace_for_call(args=args, kwargs=kwargs)
    if resolution is None:
        msg = "Unary array tracing requires one provider-backed input"
        raise TypeError(msg)
    selected_version = resolution.requested_version
    with _dynamic_trace(array_api_version=selected_version, reverse_only=True) as tape:
        traced_input = dispatch_input(args[0], name=input_name)
        traced_args = list(args)
        traced_args[0] = traced_input
        traced_output = f(*traced_args, **kwargs)
        output_treedef, output_ids = _mark_outputs(traced_output, cast("Any", tape))
        output_values = tape.values(output_ids)
        if _is_numerics_debug():
            check_tape_numerics(tape)

    return TraceResult(
        tape=tape,
        positional_specs=[],
        named_specs={},
        output_treedef=output_treedef,
        output_ids=tuple(output_ids),
        output=_concrete_output(tape, output_treedef, output_values),
        provider=provider,
        input_primals=(args[0],),
        array_api_version=selected_version,
    )


def _is_boolean_value(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, tuple):
        return bool(value) and all(_is_boolean_value(item) for item in value)
    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind == "b"
    return dtype is bool or str(dtype).rsplit(".", 1)[-1] == "bool"


def _has_tangent_space(value: object) -> bool:
    if isinstance(value, tuple):
        return any(_has_tangent_space(item) for item in value)
    if isinstance(value, (float, complex)):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and dtype_is_inexact(dtype)


def _is_locally_constant(
    answer: object,
    operands: tuple[object, ...],
    tangents: tuple[object | None, ...],
) -> bool:
    """Return whether a non-differentiable result carries no tangent.

    A boolean result, such as a comparison, has no tangent space. An integer
    result of a floating operand, such as ``argmax`` or ``argsort``, is
    piecewise constant in that operand.
    """
    if _has_tangent_space(answer):
        return False
    return _is_boolean_value(answer) or any(
        tangent is not None and _has_tangent_space(operand)
        for operand, tangent in zip(operands, tangents, strict=True)
    )


def _normalize_output_tangent(tangent: object, primal: object) -> object:
    """Represent scalar rule results through the primal output provider."""
    if hasattr(tangent, "shape") or not hasattr(primal, "shape"):
        return tangent
    if type(tangent) not in (bool, int, float, complex):
        return tangent
    namespace = _get_array_namespace(primal)
    asarray = getattr(namespace, "asarray", None) if namespace is not None else None
    return asarray(tangent) if callable(asarray) else tangent


def _run_on_trace[T](
    trace: TraceResult,
    traversal: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """Run one native traversal under the trace's Array API revision and provider."""
    with _use_array_api_version(trace.array_api_version):
        if trace.provider is None:
            return traversal(trace.tape, *args, **kwargs)
        return cast(
            "T",
            run_with_array_family_backend_provider(
                trace.provider,
                traversal,
                trace.tape,
                *args,
                **kwargs,
            ),
        )


def _node_table(node_ids: Sequence[int], values: Sequence[Any]) -> dict[int, Any]:
    return {
        node_id: value for node_id, value in zip(node_ids, values, strict=True) if value is not None
    }


def apply_jvp(
    trace: TraceResult,
    tangent_seeds: dict[int, Any],
    *,
    consume: bool = False,
) -> dict[int, Any]:
    """Apply registered traceable JVP rules in arena order."""
    tangents = _run_on_trace(
        trace,
        dynamic_jvp,
        list(tangent_seeds.items()),
        list(trace.output_ids),
        consume=consume,
    )
    return _node_table(trace.output_ids, tangents)


def apply_jvp_many(
    trace: TraceResult,
    tangent_seed_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    """Apply bounded JVP seeds in one arena traversal."""
    tangent_sets = _run_on_trace(
        trace,
        dynamic_jvp_many,
        [list(tangent_seeds.items()) for tangent_seeds in tangent_seed_sets],
        list(trace.output_ids),
    )
    return tuple(_node_table(trace.output_ids, tangents) for tangents in tangent_sets)


def _project_to_tangent_space(target_dtype: object, contribution: object) -> object:
    """Cast a contribution into the tangent space of a primal's dtype.

    Every JVP output tangent and VJP contribution passes through here, so a
    rule may return the precision or complex type of its tangents: a real
    primal receives the real part and a floating primal its own precision.
    A strong tangent of a weak Python scalar thus cannot widen a ``float32``
    result, and no primal is retained solely for its dtype.

    A value from another provider, such as an Array API cotangent pulled
    back through a Python scalar lifted with NumPy, keeps its own dtype: the
    primal's dtype object means nothing to that provider.
    """
    if isinstance(contribution, complex) and not _dtype_is_complex(target_dtype):
        return contribution.real
    contribution_dtype = getattr(contribution, "dtype", None)
    if (
        contribution_dtype is None
        or contribution_dtype is target_dtype
        or not dtype_is_inexact(target_dtype)
        or dtype_name(contribution_dtype) == dtype_name(target_dtype)
        or type(contribution_dtype).__module__.partition(".")[0]
        != type(target_dtype).__module__.partition(".")[0]
    ):
        return contribution
    projected = _astype_preserving_trace(contribution, dtype=cast("Any", target_dtype))
    # A provider scalar stays a scalar, as the transforms return rank-zero values.
    scalar_type = getattr(target_dtype, "type", None)
    if type(contribution) is getattr(contribution_dtype, "type", None) and callable(scalar_type):
        return scalar_type(projected)
    return projected


def _project_output_tangent(answer: object, tangent: object) -> object:
    """Project a JVP result, leaf by leaf for a tuple answer, onto its answer."""
    dtype = getattr(answer, "dtype", None)
    if dtype is None:
        if isinstance(answer, tuple) and isinstance(tangent, tuple):
            return tuple(
                _project_output_tangent(leaf, leaf_tangent)
                for leaf, leaf_tangent in zip(answer, tangent, strict=True)
            )
        return tangent
    if getattr(tangent, "dtype", None) is dtype:
        return tangent
    return _project_to_tangent_space(dtype, tangent)


def _apply_vjp_binding(  # noqa: PLR0913 - mirrors the native callback ABI
    *,
    op: str,
    source_location: str | None,
    rule: _TransposeRule,
    answer: object,
    operands: tuple[object, ...],
    cotangent: object,
    raw_attrs: object,
    active_positions: tuple[int, ...],
    residual: object,
    parent_specs: tuple[tuple[tuple[int, ...], object] | None, ...],
) -> list[object | None]:
    definition = rule.definition
    if definition.non_differentiable_reason is not None:
        if _is_boolean_value(answer):
            # A boolean result has no cotangent space, as in forward mode.
            return [None] * len(operands)
        msg = f"Cannot transpose primitive '{op}': it is non-differentiable"
        raise NoVJPError(
            msg,
            op=op,
            source_location=source_location,
            non_differentiable=True,
            grad_reason=definition.non_differentiable_reason,
        )

    vjp_rule = rule.vjp
    if vjp_rule is None:
        if definition.jvp is None:
            msg = (
                f"Cannot transpose primitive '{op}': no structurally validated "
                "transpose rule is installed"
            )
            raise NoVJPError(msg, op=op, source_location=source_location)
        (normalized,) = _apply_structural_vjp_binding_many(
            op=op,
            source_location=source_location,
            rule=rule,
            answer=answer,
            operands=operands,
            cotangents=(cotangent,),
            raw_attrs=raw_attrs,
            active_positions=active_positions,
            parent_specs=parent_specs,
        )
        return normalized

    attrs = _decode_rule_attrs(op, raw_attrs, operands, cotangent)
    if rule.reads_arrays:
        operands = _as_rule_arrays(operands)
        (cotangent,) = _as_rule_arrays((cotangent,))
    primals = operands if definition.vjp_needs_inputs else ()
    residual_attrs: dict[str, object] = {"residual": residual} if definition.has_residual else {}
    try:
        if rule.selective_vjp is not None:
            contributions = rule.selective_vjp(
                answer,
                *primals,
                g=cotangent,
                active_input_indices=active_positions,
                **residual_attrs,
                **attrs,
            )
        else:
            contributions = vjp_rule(
                answer,
                *primals,
                g=cotangent,
                **residual_attrs,
                **attrs,
            )
    except (AdvectError, NotImplementedError):
        raise
    except Exception as error:
        msg = f"Transpose rule for '{op}' failed: {error}"
        raise RuntimeError(msg) from error

    return _normalize_vjp_contributions(
        op=op,
        operands=operands,
        contributions=contributions,
        active_positions=active_positions,
        parent_specs=parent_specs,
    )


def _apply_structural_vjp_binding_many(  # noqa: PLR0913 - mirrors callback ABI
    *,
    op: str,
    source_location: str | None,
    rule: _TransposeRule,
    answer: object,
    operands: tuple[object, ...],
    cotangents: tuple[object, ...],
    raw_attrs: object,
    active_positions: tuple[int, ...],
    parent_specs: tuple[tuple[tuple[int, ...], object] | None, ...],
) -> tuple[list[object | None], ...]:
    definition = rule.definition
    jvp_rule = definition.jvp
    if rule.vjp is not None or jvp_rule is None:
        msg = f"Primitive '{op}' does not have a JVP-only batched transpose"
        raise NoVJPError(msg, op=op, source_location=source_location)
    attrs = _decode_rule_attrs(op, raw_attrs, operands, answer)
    if rule.reads_arrays:
        operands = _as_rule_arrays(operands)
        cotangents = _as_rule_arrays(cotangents)
    contribution_sets = transpose_jvp_structurally_many(
        op=op,
        source_location=source_location,
        answer=answer,
        primals=operands,
        attrs=attrs,
        cotangents=cotangents,
        jvp_rule=jvp_rule,
    )
    return tuple(
        _normalize_vjp_contributions(
            op=op,
            operands=operands,
            contributions=contributions,
            active_positions=active_positions,
            parent_specs=parent_specs,
        )
        for contributions in contribution_sets
    )


def _normalize_vjp_contributions(
    *,
    op: str,
    operands: tuple[object, ...],
    contributions: tuple[Any | None, ...],
    active_positions: tuple[int, ...],
    parent_specs: tuple[tuple[tuple[int, ...], object] | None, ...],
) -> list[object | None]:
    if len(contributions) > len(operands):
        msg = (
            f"Transpose rule for '{op}' returned {len(contributions)} slots "
            f"for {len(operands)} inputs"
        )
        raise RuntimeError(msg)

    normalized: list[object | None] = [None] * len(operands)
    for position in active_positions:
        raw_contribution = contributions[position] if position < len(contributions) else None
        if raw_contribution is None:
            continue
        spec = parent_specs[position]
        if spec is None:
            msg = f"Dynamic VJP for '{op}' marked literal operand {position} active"
            raise RuntimeError(msg)
        target_shape, target_dtype = spec
        contribution = raw_contribution
        contribution_shape = getattr(raw_contribution, "shape", None)
        if contribution_shape is not None and tuple(contribution_shape) != tuple(target_shape):
            contribution = _unbroadcast(raw_contribution, tuple(target_shape))
        normalized[position] = _project_to_tangent_space(target_dtype, contribution)

    return normalized


def apply_transpose(
    trace: TraceResult,
    output_cotangents: dict[int, Any],
    *,
    consume: bool = False,
) -> dict[int, Any]:
    """Run the real adjoint of one concrete linearization."""
    input_ids = trace.tape.inputs
    gradients = _run_on_trace(
        trace,
        dynamic_vjp,
        list(output_cotangents.items()),
        input_ids,
        consume=consume,
    )
    return _node_table(input_ids, gradients)


def apply_transpose_many(
    trace: TraceResult,
    output_cotangent_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    """Apply bounded cotangent seeds in one reverse arena traversal."""
    input_ids = trace.tape.inputs
    gradient_sets = _run_on_trace(
        trace,
        dynamic_vjp_many,
        [list(output_cotangents.items()) for output_cotangents in output_cotangent_sets],
        input_ids,
    )
    return tuple(_node_table(input_ids, gradients) for gradients in gradient_sets)


def apply_unary_array_pullback(trace: TraceResult, cotangent: object) -> object:
    """Apply a one-input, one-output array pullback without generic tree plumbing.

    The caller owns this invocation-local trace and drops its only reference
    immediately, making every payload unreachable without an explicit arena
    scan. Reusable ``LinearMap`` objects retain the explicit release path.
    """
    input_ids = trace.tape.inputs
    if len(input_ids) != 1 or len(trace.output_ids) != 1:
        msg = "Unary array pullback requires exactly one tape input and output"
        raise RuntimeError(msg)

    input_id = input_ids[0]
    input_value = trace.input_primals[0]
    try:
        gradients = apply_transpose(
            trace,
            {trace.output_ids[0]: cotangent},
            consume=True,
        )
        gradient = gradients.get(input_id)
        if gradient is not None:
            return gradient
        return _zeros_like(input_value)
    finally:
        trace.tape.release_payloads()


class LinearMap:
    """Reusable real-linear map captured by one concrete trace.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> _, linear = ad.linearize(lambda x: x**2, np.array([1.0, 2.0]))
    >>> with linear:
    ...     linear(np.ones(2)).tolist()
    [2.0, 4.0]
    """

    __slots__ = ("_consumed", "_single_argnum", "_trace")

    def __init__(self, trace: TraceResult, *, single_argnum: bool) -> None:
        self._trace = trace
        self._single_argnum = single_argnum
        self._consumed = False

    def _require_available(self) -> None:
        if self._consumed:
            msg = "This linearization has been closed or consumed"
            raise RuntimeError(msg)

    def _release(self) -> None:
        if self._consumed:
            return
        try:
            self._trace.tape.release_payloads()
        except BaseException:
            self._consumed = self._trace.tape.is_consumed
            raise
        self._consumed = True

    def close(self) -> None:
        """Release retained concrete values and primitive residuals."""
        self._release()

    def __enter__(self) -> Self:
        self._require_available()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _output_tangents(
        self,
        tangent_table: dict[int, Any],
        output_values: list[Any],
    ) -> Any:  # noqa: ANN401 - generic tangent pytree
        """Zero-fill disconnected output tangents and restore the output structure."""
        leaves = []
        for node_id, primal in zip(self._trace.output_ids, output_values, strict=True):
            tangent = tangent_table.get(node_id)
            if tangent is None:
                tangent = _zeros_like(primal) if hasattr(primal, "shape") else 0.0
            leaves.append(_normalize_output_tangent(tangent, primal))
        return tree_unflatten(self._trace.output_treedef, leaves)

    def _apply(
        self,
        tangents: Any,  # noqa: ANN401 - generic tangent pytree
        *,
        consume: bool,
    ) -> Any:  # noqa: ANN401 - generic tangent pytree
        self._require_available()
        try:
            tangent_seeds = _build_input_tangent_seeds(
                positional_specs=self._trace.positional_specs,
                tangents=tangents,
                single_argnum=self._single_argnum,
            )
            output_values = self._trace.tape.values(list(self._trace.output_ids))
            return self._output_tangents(
                apply_jvp(self._trace, tangent_seeds, consume=consume),
                output_values,
            )
        finally:
            if consume:
                self._release()

    def __call__(self, tangents: Any) -> Any:  # noqa: ANN401 - generic tangent pytree
        return self._unlift_outputs(self._apply(tangents, consume=False))

    def _unlift_outputs(self, value: object) -> object:
        return _unlift_scalar_tree_by_mask(
            value,
            mask=self._trace.restore_scalar_outputs,
        )

    def _apply_seed_tables_many(
        self,
        tangent_seed_sets: tuple[dict[int, Any], ...],
    ) -> tuple[Any, ...]:
        """Apply internal input-node seeds in bounded native traversals."""
        self._require_available()
        output_values = self._trace.tape.values(list(self._trace.output_ids))
        return tuple(
            self._output_tangents(tangent_table, output_values)
            for tangent_batch in batched(tangent_seed_sets, _PULLBACK_MANY_BATCH_SIZE)
            for tangent_table in apply_jvp_many(self._trace, tangent_batch)
        )

    def _output_cotangent_table(
        self,
        cotangents: Any,  # noqa: ANN401 - generic cotangent pytree
        output_values: list[Any],
    ) -> dict[int, Any]:
        """Validate one output cotangent pytree and key its leaves by output node."""
        leaves = _flatten_output_cotangents(self._trace.output_treedef, cotangents)
        return _build_grad_outputs(
            list(self._trace.output_ids),
            [
                _coerce_output_cotangent_like(cotangent, primal)
                for cotangent, primal in zip(leaves, output_values, strict=True)
            ],
        )

    def _input_gradients(self, gradients: dict[int, Any]) -> Any:  # noqa: ANN401 - gradient pytree
        """Assemble input-node gradients into the selected argument structure."""
        return _format_backward_result(
            positional_grads=[
                _build_grad_tree(spec, grads=gradients) for spec in self._trace.positional_specs
            ],
            named_grads={
                name: _build_grad_tree(spec, grads=gradients)
                for name, spec in self._trace.named_specs.items()
            },
            single_argnum=self._single_argnum,
        )

    def _pullback(
        self,
        cotangents: Any,  # noqa: ANN401 - generic cotangent pytree
        *,
        consume: bool,
    ) -> Any:  # noqa: ANN401 - generic cotangent pytree
        self._require_available()
        try:
            with _use_array_api_version(self._trace.array_api_version):
                gradients = apply_transpose(
                    self._trace,
                    self._output_cotangent_table(cotangents, tree_flatten(self._trace.output)[0]),
                    consume=consume,
                )
                return self._input_gradients(gradients)
        finally:
            if consume:
                self._release()

    def pullback(self, cotangents: Any) -> Any:  # noqa: ANN401 - generic cotangent pytree
        return self._pullback(cotangents, consume=False)

    def transpose(self) -> Callable[[Any], Any]:
        return self.pullback

    def _transpose_seed_tables_many(
        self,
        output_cotangent_sets: tuple[dict[int, Any], ...],
    ) -> tuple[dict[int, Any], ...]:
        """Apply internal output-node seeds in bounded native traversals."""
        self._require_available()
        return tuple(
            gradients
            for cotangent_batch in batched(output_cotangent_sets, _PULLBACK_MANY_BATCH_SIZE)
            for gradients in apply_transpose_many(self._trace, cotangent_batch)
        )

    def apply_many(self, tangents: tuple[Any, ...]) -> tuple[Any, ...]:
        tangent_seed_sets = tuple(
            _build_input_tangent_seeds(
                positional_specs=self._trace.positional_specs,
                tangents=tangent,
                single_argnum=self._single_argnum,
            )
            for tangent in tangents
        )
        return tuple(
            self._unlift_outputs(value) for value in self._apply_seed_tables_many(tangent_seed_sets)
        )

    def transpose_many(self, cotangents: tuple[Any, ...]) -> tuple[Any, ...]:
        self._require_available()
        output_values, _output_treedef = tree_flatten(self._trace.output)
        with _use_array_api_version(self._trace.array_api_version):
            gradient_sets = self._transpose_seed_tables_many(
                tuple(
                    self._output_cotangent_table(cotangent, output_values)
                    for cotangent in cotangents
                )
            )
            return tuple(self._input_gradients(gradients) for gradients in gradient_sets)


_STRUCTURAL_TRANSPOSE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "advect_structural_transpose_stack",
    default=(),
)

_PULLBACK_MANY_BATCH_SIZE = 16


def _zero_tangent_like(primal: object) -> object:
    if not hasattr(primal, "shape"):
        # Weak Python scalars are primal coefficients, but structural tangent
        # coordinates are ordinary provider arrays. Keeping the latter strong
        # also gives strict Array API providers one array operand in local JVP
        # formulas that combine a scalar partial with a scalar tangent.
        return xp.asarray(0.0)
    return xp.zeros_like(primal)


def transpose_jvp_structurally_many(
    *,
    op: str,
    source_location: str | None,
    answer: object,
    primals: tuple[object, ...],
    attrs: dict[str, Any],
    cotangents: tuple[object, ...],
    jvp_rule: Callable[..., object],
) -> tuple[tuple[Any | None, ...], ...]:
    """Trace one JVP and transpose a bounded cotangent group through it."""
    with _structural_jvp_linear_map(
        op=op,
        source_location=source_location,
        answer=answer,
        primals=primals,
        attrs=attrs,
        jvp_rule=jvp_rule,
    ) as tangent_linear:
        contribution_sets = tangent_linear.transpose_many(cotangents)
    if not all(isinstance(contributions, tuple) for contributions in contribution_sets):
        msg = f"Structural transpose for '{op}' returned an invalid pullback structure"
        raise TypeError(msg)
    return cast("tuple[tuple[Any | None, ...], ...]", contribution_sets)


def _materialize_internal_complex_scalars(value: object) -> object:
    """Represent internal complex coefficients as backend 0-D arrays."""
    leaves, treedef = tree_flatten(value)
    if not any(isinstance(leaf, complex) for leaf in leaves):
        return value
    return tree_unflatten(
        treedef,
        [xp.asarray(leaf) if isinstance(leaf, complex) else leaf for leaf in leaves],
    )


@contextmanager
def _structural_jvp_linear_map(
    *,
    op: str,
    source_location: str | None,
    answer: object,
    primals: tuple[object, ...],
    attrs: dict[str, Any],
    jvp_rule: Callable[..., object],
) -> Generator[LinearMap, None, None]:
    """Build one reusable tangent map for a structural JVP transpose."""
    stack = _STRUCTURAL_TRANSPOSE_STACK.get()
    if op in stack:
        cycle = " -> ".join((*stack, op))
        msg = (
            f"Cannot structurally transpose JVP rule for '{op}': the rule "
            f"depends recursively on a primitive without an explicit transpose "
            f"({cycle}). Register a transpose for the linear primitive basis."
        )
        raise NoVJPError(msg, op=op, source_location=source_location)

    stack_token = _STRUCTURAL_TRANSPOSE_STACK.set((*stack, op))
    try:
        jvp_trace = _trace_structural_jvp(
            jvp_rule=jvp_rule,
            answer=answer,
            primals=primals,
            attrs=attrs,
        )
        # Validate real linearity and keep the values depending on tangent inputs.
        tangent_dependent_ids = jvp_trace.tape.analyze_real_linearity(
            [
                leaf_spec.node_id
                for spec in jvp_trace.positional_specs
                for leaf_spec in spec.leaf_specs
                if leaf_spec.node_id is not None
            ],
            op,
        )
        # The JVP trace lifts answers and primals as inputs so nested
        # differentiation can retain their enclosing-trace dependencies. They
        # are coefficients of the linear map, however, not transpose targets.
        # Restrict the local reverse sweep to values that structurally depend
        # on tangent inputs so it does not differentiate irrelevant primal-only
        # branches.
        jvp_trace.tape.set_active_nodes(list(tangent_dependent_ids))
        with LinearMap(jvp_trace, single_argnum=False) as tangent_linear:
            yield tangent_linear
    finally:
        _STRUCTURAL_TRANSPOSE_STACK.reset(stack_token)


def _trace_structural_jvp(
    *,
    jvp_rule: Callable[..., object],
    answer: object,
    primals: tuple[object, ...],
    attrs: dict[str, Any],
) -> TraceResult:
    """Trace a JVP rule with its answer, primals, and zero tangents as inputs.

    Every operand is an input, so this skips the signature-based selection of
    ``trace_call``. The answer and primals stay lifted so nested
    differentiation retains their enclosing-trace dependencies, but only the
    tangent input specifications are returned as transpose targets.
    """
    arity = len(primals)
    operands = tuple(
        _materialize_internal_complex_scalars(operand)
        for operand in (answer, *primals, *(_zero_tangent_like(primal) for primal in primals))
    )

    def trace_operands_and_rule(
        tape: DynamicTape,
        xp: object | None,
    ) -> tuple[object, list[Any], dict[str, Any]]:
        traced_operands: list[object] = []
        specs: list[Any] = []
        for index, operand in enumerate(operands):
            traced, spec = _trace_value_as_inputs(
                tape,
                operand,
                prefix=f"operands[{index}]",
                xp=xp,
            )
            traced_operands.append(traced)
            specs.append(spec)
        traced_output = jvp_rule(
            traced_operands[0],
            *traced_operands[1 : arity + 1],
            tangents=tuple(traced_operands[arity + 1 :]),
            **attrs,
        )
        return traced_output, specs[arity + 1 :], {}

    return _trace(trace_operands_and_rule, args=operands, kwargs={})


def linearize_call(  # noqa: PLR0913 - forwards every trace_call option
    f: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argnums: tuple[int, ...],
    argnames: tuple[str, ...] | None,
    single_argnum: bool,
    reverse_only: bool = False,
    require_jvp: bool = False,
    select_output: Callable[[Any], Any] | None = None,
) -> tuple[Any, LinearMap]:
    trace = trace_call(
        f,
        args=args,
        kwargs=kwargs,
        argnums=argnums,
        argnames=argnames,
        reverse_only=reverse_only,
        require_jvp=require_jvp,
        select_output=select_output,
    )
    return trace.output, LinearMap(trace, single_argnum=single_argnum)


__all__ = [
    "LinearMap",
    "TraceResult",
    "apply_jvp",
    "apply_transpose",
    "linearize_call",
    "trace_call",
]
