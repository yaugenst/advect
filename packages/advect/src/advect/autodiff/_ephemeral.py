"""Concrete define-by-run linearization on the lightweight SSA tape."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import batched
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
)
from advect.autodiff.api.trace import _mark_outputs
from advect.autodiff.rules.array_family._backend_runtime import (
    _maybe_unwrap_array_family_jvp_rule,
    _maybe_unwrap_array_family_vjp_rule,
    run_with_array_family_backend_provider,
    xp,
)
from advect.autodiff.rules.array_family.providers import (
    try_resolve_array_family_backend_provider,
)
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.core._array_api.providers import (
    _get_array_namespace,
    _negotiate_array_namespace_for_call,
)
from advect.core._backends import dispatch_input, get_hook
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
    from collections.abc import Callable, Generator

    from advect.autodiff.rules.array_family.providers import ArrayFamilyBackendProvider
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


_BATCH_VJP_ATTR = "__advect_vjp_many__"


def _transpose_rule(op: str) -> _TransposeRule:
    definition = get_registry()._get_canonical(op)  # noqa: SLF001 - tape IDs are canonical
    vjp = definition.vjp
    if vjp is not None:
        vjp = _maybe_unwrap_array_family_vjp_rule(vjp) or vjp
    selective = None if vjp is None else getattr(vjp, "__advect_vjp_for_input_indices__", None)
    return _TransposeRule(
        definition=definition,
        vjp=vjp,
        selective_vjp=(
            cast("Callable[..., tuple[Any | None, ...]]", selective)
            if callable(selective)
            else None
        ),
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


def _jvp_binding(op: str) -> Callable[..., object]:
    cache = _dynamic_binding_cache()
    cached = cache.jvp.get(op)
    if cached is not None:
        return cached
    definition = get_registry()._get_canonical(op)  # noqa: SLF001 - native IDs are canonical
    rule = definition.jvp
    active_rule = None
    if rule is not None:
        active_rule = _maybe_unwrap_array_family_jvp_rule(rule) or rule

    def apply(
        answer: object,
        operands: tuple[object, ...],
        tangents: tuple[object | None, ...],
        raw_attrs: object,
        source_location: str | None,
    ) -> object:
        if active_rule is None:
            if definition.non_differentiable_reason is not None and _is_boolean_value(answer):
                return None
            msg = f"Cannot linearize primitive '{op}': no JVP rule is installed"
            raise NoJVPError(msg, op=op, source_location=source_location)
        attrs = _decode_attrs_for_vjp(op, cast("dict[str, Any]", raw_attrs))
        with _numerics_context("JVP propagation", source_location):
            return _checked_numerics(
                active_rule(answer, *operands, tangents=tangents, **attrs),
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
        definition = registry._get_canonical(op)  # noqa: SLF001 - canonical tape ID
        jvp_bindings.append(binding_cache.jvp.get(op) or _jvp_binding(op))
        vjp_bindings.append(binding_cache.vjp.get(op) or _vjp_binding(op))
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
) -> Generator[DynamicTape, None, None]:
    tape = DynamicTape()
    _set_active_recorder(
        cast("Any", tape),
        trace_kind="autodiff_dynamic",
        array_api_version=array_api_version,
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
) -> TraceResult:
    """Trace one concrete call without constructing a durable graph."""
    resolution = _negotiate_array_namespace_for_call(args=args, kwargs=kwargs)
    if resolution is None:
        default_namespace = get_hook("advect.default_array_namespace")
        namespace = None if default_namespace is None else default_namespace()
        asarray = getattr(namespace, "asarray", None)
        if callable(asarray):
            sample = asarray(0.0)
            resolution = _negotiate_array_namespace_for_call(args=(sample,), kwargs={})
    selected_version = (
        LATEST_ARRAY_API_VERSION if resolution is None else resolution.requested_version
    )
    with _dynamic_trace(
        array_api_version=selected_version,
        reverse_only=reverse_only,
    ) as tape:
        normalized_argnums = _normalize_argnums_for_call(argnums, nargs=len(args))
        xp = None if resolution is None else resolution.raw_namespace
        (
            traced_args,
            traced_kwargs,
            positional_specs,
            named_specs,
        ) = _trace_selected_args_and_kwargs(
            f,
            graph=cast("Any", tape),
            args=args,
            kwargs=kwargs,
            normalized_argnums=normalized_argnums,
            argnames=argnames,
            xp=xp,
        )
        traced_output = f(*traced_args, **traced_kwargs)
        output_treedef, output_ids = _mark_outputs(traced_output, cast("Any", tape))
        output_values = tape.values(output_ids)
        if _is_numerics_debug():
            check_tape_numerics(tape)
        try:
            provider = try_resolve_array_family_backend_provider(
                *tape.values(tape.inputs),
                *output_values,
                array_api_version=selected_version,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            provider = None

    try:
        concrete_output = (
            output_values[0]
            if output_treedef.node_type is None
            else tree_unflatten(output_treedef, output_values)
        )
    except Exception:
        tape.release_payloads()
        raise
    restore_scalar_outputs = tuple(tape.weak_mask(output_ids))
    return TraceResult(
        tape=tape,
        positional_specs=positional_specs,
        named_specs=named_specs,
        output_treedef=output_treedef,
        output_ids=tuple(output_ids),
        output=concrete_output,
        provider=provider,
        restore_scalar_outputs=restore_scalar_outputs,
        array_api_version=selected_version,
    )


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

    try:
        concrete_output = (
            output_values[0]
            if output_treedef.node_type is None
            else tree_unflatten(output_treedef, output_values)
        )
    except Exception:
        tape.release_payloads()
        raise
    return TraceResult(
        tape=tape,
        positional_specs=[],
        named_specs={},
        output_treedef=output_treedef,
        output_ids=tuple(output_ids),
        output=concrete_output,
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
    return bool(
        getattr(dtype, "kind", None) == "b"
        or getattr(dtype, "name", None) == "bool"
        or dtype is bool
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


def _apply_jvp(
    trace: TraceResult,
    tangent_seeds: dict[int, Any],
    *,
    consume: bool,
) -> dict[int, Any]:
    tangents = dynamic_jvp(
        trace.tape,
        list(tangent_seeds.items()),
        list(trace.output_ids),
        consume=consume,
    )
    return {
        node_id: tangent
        for node_id, tangent in zip(trace.output_ids, tangents, strict=True)
        if tangent is not None
    }


def _trace_provider(trace: TraceResult) -> ArrayFamilyBackendProvider | None:
    return trace.provider


def apply_jvp(
    trace: TraceResult,
    tangent_seeds: dict[int, Any],
    *,
    consume: bool = False,
) -> dict[int, Any]:
    """Apply registered traceable JVP rules in arena order."""
    with _use_array_api_version(trace.array_api_version):
        provider = _trace_provider(trace)
        if provider is None:
            return _apply_jvp(trace, tangent_seeds, consume=consume)
        return cast(
            "dict[int, Any]",
            run_with_array_family_backend_provider(
                provider,
                _apply_jvp,
                trace,
                tangent_seeds,
                consume=consume,
            ),
        )


def _apply_jvp_many(
    trace: TraceResult,
    tangent_seed_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    tangent_sets = dynamic_jvp_many(
        trace.tape,
        [list(tangent_seeds.items()) for tangent_seeds in tangent_seed_sets],
        list(trace.output_ids),
    )
    return tuple(
        {
            node_id: tangent
            for node_id, tangent in zip(trace.output_ids, tangents, strict=True)
            if tangent is not None
        }
        for tangents in tangent_sets
    )


def apply_jvp_many(
    trace: TraceResult,
    tangent_seed_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    """Apply bounded JVP seeds in one arena traversal."""
    with _use_array_api_version(trace.array_api_version):
        provider = _trace_provider(trace)
        if provider is None:
            return _apply_jvp_many(trace, tangent_seed_sets)
        return cast(
            "tuple[dict[int, Any], ...]",
            run_with_array_family_backend_provider(
                provider,
                _apply_jvp_many,
                trace,
                tangent_seed_sets,
            ),
        )


def _dtype_is_complex(dtype: object) -> bool:
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind == "c"
    return "complex" in str(dtype).lower()


def _real_part(value: object) -> object:
    namespace = _get_array_namespace(value)
    if namespace is not None and hasattr(namespace, "real"):
        return namespace.real(value)
    real = getattr(value, "real", None)
    return real if real is not None else value


def _project_cotangent_to_dtype(target_dtype: object, contribution: object) -> object:
    """Project without retaining a primal value solely for its dtype."""
    if isinstance(contribution, tuple):
        return contribution
    contribution_dtype = getattr(contribution, "dtype", None)
    if not _dtype_is_complex(target_dtype) and (
        isinstance(contribution, complex)
        or (contribution_dtype is not None and _dtype_is_complex(contribution_dtype))
    ):
        return _real_part(contribution)
    return contribution


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
        msg = f"Cannot transpose primitive '{op}': it is non-differentiable"
        raise NoVJPError(
            msg,
            op=op,
            source_location=source_location,
            non_differentiable=True,
            grad_reason=definition.non_differentiable_reason,
        )

    attrs = (
        {} if raw_attrs is None else _decode_attrs_for_vjp(op, cast("dict[str, Any]", raw_attrs))
    )
    vjp_rule = rule.vjp
    if vjp_rule is None:
        jvp_rule = definition.jvp
        if jvp_rule is None:
            msg = (
                f"Cannot transpose primitive '{op}': no structurally validated "
                "transpose rule is installed"
            )
            raise NoVJPError(msg, op=op, source_location=source_location)
        contributions = transpose_jvp_structurally(
            op=op,
            source_location=source_location,
            answer=answer,
            primals=operands,
            attrs=attrs,
            cotangent=cotangent,
            jvp_rule=jvp_rule,
        )
    else:
        primals = operands if definition.vjp_needs_inputs else ()
        residual_attrs: dict[str, object] = (
            {"residual": residual} if definition.has_residual else {}
        )
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
    attrs = (
        {} if raw_attrs is None else _decode_attrs_for_vjp(op, cast("dict[str, Any]", raw_attrs))
    )
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
        contribution_dtype = getattr(contribution, "dtype", None)
        needs_projection = contribution_dtype != target_dtype and (
            isinstance(contribution, (complex, tuple))
            or (contribution_dtype is not None and _dtype_is_complex(contribution_dtype))
        )
        normalized[position] = (
            cast("Any", _project_cotangent_to_dtype(target_dtype, contribution))
            if needs_projection
            else contribution
        )

    return normalized


def _apply_transpose(
    trace: TraceResult,
    output_cotangents: dict[int, Any],
    *,
    consume: bool,
) -> dict[int, Any]:
    input_ids = trace.tape.inputs
    gradients = dynamic_vjp(
        trace.tape,
        list(output_cotangents.items()),
        input_ids,
        consume=consume,
    )
    return {
        node_id: gradient
        for node_id, gradient in zip(input_ids, gradients, strict=True)
        if gradient is not None
    }


def apply_transpose(
    trace: TraceResult,
    output_cotangents: dict[int, Any],
    *,
    consume: bool = False,
) -> dict[int, Any]:
    """Run the real adjoint of one concrete linearization."""
    with _use_array_api_version(trace.array_api_version):
        provider = _trace_provider(trace)
        if provider is None:
            return _apply_transpose(
                trace,
                output_cotangents,
                consume=consume,
            )
        return cast(
            "dict[int, Any]",
            run_with_array_family_backend_provider(
                provider,
                _apply_transpose,
                trace,
                output_cotangents,
                consume=consume,
            ),
        )


def _apply_transpose_many(
    trace: TraceResult,
    output_cotangent_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    input_ids = trace.tape.inputs
    gradient_sets = dynamic_vjp_many(
        trace.tape,
        [list(output_cotangents.items()) for output_cotangents in output_cotangent_sets],
        input_ids,
    )
    return tuple(
        {
            node_id: gradient
            for node_id, gradient in zip(input_ids, gradients, strict=True)
            if gradient is not None
        }
        for gradients in gradient_sets
    )


def apply_transpose_many(
    trace: TraceResult,
    output_cotangent_sets: tuple[dict[int, Any], ...],
) -> tuple[dict[int, Any], ...]:
    """Apply bounded cotangent seeds in one reverse arena traversal."""
    with _use_array_api_version(trace.array_api_version):
        provider = _trace_provider(trace)
        if provider is None:
            return _apply_transpose_many(trace, output_cotangent_sets)
        return cast(
            "tuple[dict[int, Any], ...]",
            run_with_array_family_backend_provider(
                provider,
                _apply_transpose_many,
                trace,
                output_cotangent_sets,
            ),
        )


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
            tangent_table = apply_jvp(self._trace, tangent_seeds, consume=consume)
            leaves = []
            for node_id, primal in zip(self._trace.output_ids, output_values, strict=True):
                tangent = tangent_table.get(node_id)
                if tangent is None:
                    tangent = _zeros_like(primal) if hasattr(primal, "shape") else 0.0
                leaves.append(_normalize_output_tangent(tangent, primal))
            return tree_unflatten(self._trace.output_treedef, leaves)
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
        if not tangent_seed_sets:
            return ()
        output_values = self._trace.tape.values(list(self._trace.output_ids))
        results: list[Any] = []
        for tangent_batch in batched(tangent_seed_sets, _PULLBACK_MANY_BATCH_SIZE):
            tangent_tables = apply_jvp_many(self._trace, tangent_batch)
            for tangent_table in tangent_tables:
                leaves = []
                for node_id, primal in zip(
                    self._trace.output_ids,
                    output_values,
                    strict=True,
                ):
                    tangent = tangent_table.get(node_id)
                    if tangent is None:
                        tangent = _zeros_like(primal) if hasattr(primal, "shape") else 0.0
                    leaves.append(_normalize_output_tangent(tangent, primal))
                results.append(tree_unflatten(self._trace.output_treedef, leaves))
        return tuple(results)

    def _consume(
        self,
        tangents: Any,  # noqa: ANN401 - generic tangent pytree
    ) -> Any:  # noqa: ANN401 - generic tangent pytree
        return self._apply(tangents, consume=True)

    def _pullback(
        self,
        cotangents: Any,  # noqa: ANN401 - generic cotangent pytree
        *,
        consume: bool,
    ) -> Any:  # noqa: ANN401 - generic cotangent pytree
        self._require_available()
        try:
            with _use_array_api_version(self._trace.array_api_version):
                leaves = _flatten_output_cotangents(self._trace.output_treedef, cotangents)
                output_values, _output_treedef = tree_flatten(self._trace.output)
                normalized_leaves = [
                    _coerce_output_cotangent_like(cotangent, primal)
                    for cotangent, primal in zip(leaves, output_values, strict=True)
                ]
                output_cotangents = _build_grad_outputs(
                    list(self._trace.output_ids),
                    normalized_leaves,
                )
                gradients = apply_transpose(
                    self._trace,
                    output_cotangents,
                    consume=consume,
                )
                positional = [
                    _build_grad_tree(spec, grads=gradients) for spec in self._trace.positional_specs
                ]
                named = {
                    name: _build_grad_tree(spec, grads=gradients)
                    for name, spec in self._trace.named_specs.items()
                }
                return _format_backward_result(
                    positional_grads=positional,
                    named_grads=named,
                    single_argnum=self._single_argnum,
                )
        finally:
            if consume:
                self._release()

    def pullback(self, cotangents: Any) -> Any:  # noqa: ANN401 - generic cotangent pytree
        return self._pullback(cotangents, consume=False)

    def _consume_pullback(
        self,
        cotangents: Any,  # noqa: ANN401 - generic cotangent pytree
    ) -> Any:  # noqa: ANN401 - generic cotangent pytree
        return self._pullback(cotangents, consume=True)

    def transpose(self) -> Callable[[Any], Any]:
        return self.pullback

    def _transpose_seed_tables_many(
        self,
        output_cotangent_sets: tuple[dict[int, Any], ...],
    ) -> tuple[dict[int, Any], ...]:
        """Apply internal output-node seeds in bounded native traversals."""
        self._require_available()
        results: list[dict[int, Any]] = []
        for cotangent_batch in batched(
            output_cotangent_sets,
            _PULLBACK_MANY_BATCH_SIZE,
        ):
            results.extend(apply_transpose_many(self._trace, cotangent_batch))
        return tuple(results)

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
        results: list[Any] = []
        for cotangent_batch in batched(cotangents, _PULLBACK_MANY_BATCH_SIZE):
            output_cotangent_sets = tuple(
                _build_grad_outputs(
                    list(self._trace.output_ids),
                    _flatten_output_cotangents(self._trace.output_treedef, cotangent),
                )
                for cotangent in cotangent_batch
            )
            gradient_sets = self._transpose_seed_tables_many(output_cotangent_sets)
            for gradients in gradient_sets:
                positional = [
                    _build_grad_tree(spec, grads=gradients) for spec in self._trace.positional_specs
                ]
                named = {
                    name: _build_grad_tree(spec, grads=gradients)
                    for name, spec in self._trace.named_specs.items()
                }
                results.append(
                    _format_backward_result(
                        positional_grads=positional,
                        named_grads=named,
                        single_argnum=self._single_argnum,
                    )
                )
        return tuple(results)


_STRUCTURAL_TRANSPOSE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "advect_structural_transpose_stack",
    default=(),
)

_PULLBACK_MANY_BATCH_SIZE = 16


def validate_real_linear_trace(
    trace: TraceResult,
    *,
    tangent_input_ids: frozenset[int],
    primitive_name: str,
) -> frozenset[int]:
    """Validate a JVP and return the values that depend on tangent inputs."""
    dependent_ids = trace.tape.analyze_real_linearity(
        list(tangent_input_ids),
        primitive_name,
    )
    return frozenset(dependent_ids)


def _zero_tangent_like(primal: object) -> object:
    if not hasattr(primal, "shape"):
        # Weak Python scalars are primal coefficients, but structural tangent
        # coordinates are ordinary provider arrays. Keeping the latter strong
        # also gives strict Array API providers one array operand in local JVP
        # formulas that combine a scalar partial with a scalar tangent.
        return xp.asarray(0.0)
    return xp.zeros_like(primal)


def _spec_node_ids(specs: list[Any]) -> frozenset[int]:
    return frozenset(
        leaf_spec.node_id
        for spec in specs
        for leaf_spec in spec.leaf_specs
        if leaf_spec.node_id is not None
    )


def transpose_jvp_structurally(
    *,
    op: str,
    source_location: str | None,
    answer: object,
    primals: tuple[object, ...],
    attrs: dict[str, Any],
    cotangent: object,
    jvp_rule: Callable[..., object],
) -> tuple[Any | None, ...]:
    """Trace, validate, and transpose a JVP without numerical basis probing."""
    with _structural_jvp_linear_map(
        op=op,
        source_location=source_location,
        answer=answer,
        primals=primals,
        attrs=attrs,
        jvp_rule=jvp_rule,
    ) as tangent_linear:
        tangent_contributions = tangent_linear.pullback(cotangent)
    if not isinstance(tangent_contributions, tuple):
        msg = f"Structural transpose for '{op}' returned an invalid pullback structure"
        raise TypeError(msg)
    return cast("tuple[Any | None, ...]", tuple(tangent_contributions))


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
    tangent_zeros = tuple(_zero_tangent_like(primal) for primal in primals)

    def jvp_program(*operands: object) -> object:
        runtime_answer = operands[0]
        arity = len(primals)
        runtime_primals = operands[1 : arity + 1]
        runtime_tangents = operands[arity + 1 :]
        return jvp_rule(
            runtime_answer,
            *runtime_primals,
            tangents=runtime_tangents,
            **attrs,
        )

    operands = tuple(
        _materialize_internal_complex_scalars(operand)
        for operand in (answer, *primals, *tangent_zeros)
    )
    all_argnums = tuple(range(len(operands)))
    try:
        jvp_trace = trace_call(
            jvp_program,
            args=operands,
            kwargs={},
            argnums=all_argnums,
            argnames=None,
            reverse_only=False,
        )
        tangent_specs = jvp_trace.positional_specs[1 + len(primals) :]
        tangent_dependent_ids = validate_real_linear_trace(
            jvp_trace,
            tangent_input_ids=_spec_node_ids(tangent_specs),
            primitive_name=op,
        )
        # The JVP trace lifts answers and primals as inputs so nested
        # differentiation can retain their enclosing-trace dependencies. They
        # are coefficients of the linear map, however, not transpose targets.
        # Restrict the local reverse sweep to values that structurally depend
        # on tangent inputs so it does not differentiate irrelevant primal-only
        # branches.
        jvp_trace.tape.set_active_nodes(list(tangent_dependent_ids))
        tangent_trace = replace(jvp_trace, positional_specs=tangent_specs)
        with LinearMap(tangent_trace, single_argnum=False) as tangent_linear:
            yield tangent_linear
    finally:
        _STRUCTURAL_TRANSPOSE_STACK.reset(stack_token)


def linearize_call(
    f: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argnums: tuple[int, ...],
    argnames: tuple[str, ...] | None,
    single_argnum: bool,
    reverse_only: bool = False,
) -> tuple[Any, LinearMap]:
    trace = trace_call(
        f,
        args=args,
        kwargs=kwargs,
        argnums=argnums,
        argnames=argnames,
        reverse_only=reverse_only,
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
