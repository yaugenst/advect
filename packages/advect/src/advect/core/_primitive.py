# ruff: noqa: ANN401  # Public primitive rules are backend-generic.
"""Unified user-authored primitive contract."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, cast, overload

from advect.core._context import (
    _active_trace_requires_jvp,
    _get_active_recorder,
    _get_active_trace_kind,
    _is_rematerializing,
    is_tracing,
)
from advect.core._errors import NoJVPError, TracingError
from advect.core._primitive_call import (
    _flatten_input_gradients,
    _flatten_primitive_output,
    _infer_namespace,
    _normalize_output_pytree,
    _reconstruct_primitive_call,
    _reconstruct_primitive_output,
    _split_primitive_attrs,
    _validate_output_treedef,
    trace_primitive_call,
)
from advect.core._protocols import _snapshot_traced
from advect.core._pytree import _tree_contains_tracer, tree_flatten
from advect.core._registry import get_registry
from advect.core._registry_types import OpDef
from advect.core._residual import _normalize_primitive_execution

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from advect.core._residual import _PrimitiveExecution


class MissingPrimitiveRuleError(TracingError):
    """Raised when a primitive lacks a rule required by a transform."""


_RULE_SENTINEL = object()
_SELECT_INPUTS_VJP_ATTR = "__advect_vjp_for_input_indices__"


def _normalize_name(name: str) -> str:
    stripped = name.removeprefix("custom.")
    if not stripped or stripped.startswith("advect."):
        msg = "Custom primitive names must be non-empty and outside the reserved 'advect' namespace"
        raise ValueError(msg)
    return f"custom.{stripped}"


def _normalize_argnames(names: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(names)
    if any(not isinstance(name, str) or not name for name in normalized):
        msg = f"{label} must contain non-empty strings"
        raise TypeError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"{label} must not contain duplicates"
        raise ValueError(msg)
    return normalized


def _callable_signature(function: Callable[..., Any], *, label: str) -> inspect.Signature:
    try:
        return inspect.signature(function)
    except (TypeError, ValueError) as error:
        msg = f"Cannot inspect the signature of {label}"
        raise TypeError(msg) from error


def _implementation_signature(
    function: Callable[..., Any],
    *,
    primitive_name: str,
    declared_argnames: frozenset[str],
) -> inspect.Signature:
    signature = _callable_signature(function, label=f"primitive {primitive_name!r}")
    unsupported = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    invalid = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind in unsupported
    ]
    if invalid:
        msg = (
            f"Primitive '{primitive_name}' signatures must use named, fixed parameters; "
            f"unsupported parameters: {', '.join(invalid)}"
        )
        raise TypeError(msg)
    unknown = declared_argnames.difference(signature.parameters)
    if unknown:
        joined = ", ".join(sorted(unknown))
        msg = f"Primitive '{primitive_name}' declares unknown argument name(s): {joined}"
        raise ValueError(msg)
    return signature


def _contains_tracer(value: Any) -> bool:
    return _tree_contains_tracer(value)


def _contains_active_tracer(value: Any) -> bool:
    recorder = _get_active_recorder()
    if recorder is None:
        return False
    leaves, _treedef = tree_flatten(value)
    return any(
        getattr(leaf, "recorder", None) is recorder
        and recorder.node_is_active(_snapshot_traced(leaf)[0])
        for leaf in leaves
        if callable(getattr(leaf, "_advect_snapshot", None))
    )


class Primitive[**P, R]:
    """Callable authoring handle returned by ``advect.primitive``.

    The handle preserves the implementation's signature and exposes
    :meth:`def_abstract`, :meth:`def_jvp`, and :meth:`def_transpose` for
    attaching rules to the same canonical ``custom.*`` operation. It is not a
    separately constructed or registered public type.
    """

    def __init__(
        self,
        name: str,
        implementation: Callable[P, R],
        *,
        static_argnames: tuple[str, ...] = (),
        nondiff_argnames: tuple[str, ...] = (),
        residual: bool = False,
        variable_output_arity: bool = False,
    ) -> None:
        if type(residual) is not bool:
            msg = "residual must be a boolean"
            raise TypeError(msg)
        if type(variable_output_arity) is not bool:
            msg = "variable_output_arity must be a boolean"
            raise TypeError(msg)
        op_name = _normalize_name(name)
        static_names = _normalize_argnames(static_argnames, label="static_argnames")
        nondiff_names = _normalize_argnames(nondiff_argnames, label="nondiff_argnames")
        overlap = set(static_names).intersection(nondiff_names)
        if overlap:
            joined = ", ".join(sorted(overlap))
            msg = f"Primitive arguments cannot be both static and nondifferentiable: {joined}"
            raise ValueError(msg)
        signature = _implementation_signature(
            implementation,
            primitive_name=op_name.removeprefix("custom."),
            declared_argnames=frozenset((*static_names, *nondiff_names)),
        )

        registry = get_registry()
        if registry.has(op_name):
            msg = f"Operation identity '{op_name}' is already registered"
            raise ValueError(msg)
        registry.register(
            OpDef(
                name=op_name,
                output_arity_known=False,
                static_argnames=static_names,
                nondiff_argnames=nondiff_names,
                has_residual=residual,
                variable_output_arity=variable_output_arity,
                implementation=cast("Callable[..., Any]", implementation),
                signature=signature,
            )
        )
        self._op_name = op_name
        functools.update_wrapper(self, implementation)

    @classmethod
    def _linked(cls, op_name: str) -> Primitive[Any, Any]:
        """Create a handle for an already-registered custom operation."""
        linked = cls.__new__(cls)
        linked._op_name = op_name  # noqa: SLF001 - alternate constructor
        implementation = cast("Callable[..., Any]", get_registry().get(op_name).implementation)
        functools.update_wrapper(linked, implementation)
        return cast("Primitive[Any, Any]", linked)

    @property
    def _definition(self) -> OpDef:
        return get_registry().get(self._op_name)

    @property
    def name(self) -> str:
        return self._op_name.removeprefix("custom.")

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def static_argnames(self) -> tuple[str, ...]:
        return self._definition.static_argnames

    @property
    def nondiff_argnames(self) -> tuple[str, ...]:
        return self._definition.nondiff_argnames

    @property
    def has_residual(self) -> bool:
        return self._definition.has_residual

    @property
    def _signature(self) -> inspect.Signature | None:
        return self._definition.signature

    @property
    def _abstract_rule(self) -> Callable[..., Any] | None:
        return self._definition.abstract_rule

    @property
    def _jvp_rule(self) -> Callable[..., Any] | None:
        rule = self._definition.jvp
        return None if rule is None else cast("Callable[..., Any]", inspect.unwrap(rule))

    @property
    def _transpose_rule(self) -> Callable[..., Any] | None:
        rule = self._definition.vjp
        return None if rule is None else cast("Callable[..., Any]", inspect.unwrap(rule))

    @property
    def _dynamic_argnames(self) -> tuple[str, ...]:
        signature = self._require_signature()
        static = set(self.static_argnames)
        return tuple(name for name in signature.parameters if name not in static)

    def _require_signature(self) -> inspect.Signature:
        signature = self._signature
        if signature is None:
            msg = "Linked primitive is missing its implementation signature"
            raise AssertionError(msg)
        return signature

    def _validate_abstract_signature(
        self,
        function: Callable[..., Any],
        *,
        implementation_signature: inspect.Signature | None = None,
    ) -> None:
        if implementation_signature is None:
            implementation_signature = self._signature
        if implementation_signature is None:
            return
        rule_signature = _callable_signature(
            function,
            label=f"abstract rule for primitive {self.name!r}",
        )
        arguments = dict.fromkeys(implementation_signature.parameters, _RULE_SENTINEL)
        try:
            rule_signature.bind(**arguments)
        except TypeError as error:
            msg = (
                f"Primitive '{self.name}' abstract rule must accept the implementation "
                f"arguments by name: {error}"
            )
            raise TypeError(msg) from error

    def _validate_derivative_signature(
        self,
        function: Callable[..., Any],
        *,
        rule: str,
        positional_count: int,
    ) -> None:
        signature = _callable_signature(
            function,
            label=f"{rule} rule for primitive {self.name!r}",
        )
        static = dict.fromkeys(self.static_argnames, _RULE_SENTINEL)
        try:
            signature.bind(*((_RULE_SENTINEL,) * positional_count), **static)
        except TypeError as error:
            msg = (
                f"Primitive '{self.name}' {rule} rule must accept its rule inputs and "
                f"declared static arguments by name: {error}"
            )
            raise TypeError(msg) from error

    def _bind_call(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        signature = self._require_signature()
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as error:
            msg = f"Invalid call to primitive '{self.name}': {error}"
            raise TypeError(msg) from error
        bound.apply_defaults()
        return dict(bound.arguments)

    def _dispatch_exact(self, *args: Any, **kwargs: Any) -> _PrimitiveExecution:
        arguments = self._bind_call(args, kwargs)
        return self._dispatch_normalized(arguments)

    def _dispatch_normalized(
        self,
        arguments: Mapping[str, Any],
    ) -> _PrimitiveExecution:
        """Execute one normalized primitive call."""
        implementation = cast("Callable[..., Any]", self._definition.implementation)
        return _normalize_primitive_execution(
            implementation(**arguments),
            primitive_name=self.name,
            has_residual=self.has_residual,
        )

    def _dispatch_impl(self, *args: Any, **kwargs: Any) -> Any:
        with self._dispatch_exact(*args, **kwargs) as execution:
            return execution.output

    def _partition_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        static_arguments = {name: arguments[name] for name in self.static_argnames}
        for name, value in static_arguments.items():
            if _contains_tracer(value):
                msg = (
                    f"Primitive '{self.name}' argument '{name}' is declared static but "
                    "received a traced value. Pass concrete configuration data or remove "
                    "the argument from static_argnames."
                )
                raise TypeError(msg)
        return static_arguments, {
            name: value for name, value in arguments.items() if name not in static_arguments
        }

    def _trace_dynamic_call(
        self,
        arguments: Mapping[str, Any],
        *,
        track_output_arity: bool,
    ) -> Any:
        """Record one concrete-only call, optionally allowing per-node output arity."""
        if _active_trace_requires_jvp() and self._jvp_rule is None:
            differentiable = set(arguments).difference(
                self.static_argnames,
                self.nondiff_argnames,
            )
            if any(_contains_active_tracer(arguments[name]) for name in differentiable):
                msg = f"Cannot linearize primitive '{self.op_name}': no JVP rule is installed"
                raise NoJVPError(msg, op=self.op_name)
        static_arguments, dynamic_arguments = self._partition_arguments(arguments)
        recorder = _get_active_recorder()
        if recorder is None:
            msg = "Primitive tracing requires an active trace"
            raise RuntimeError(msg)
        from advect.core._stage import call_primitive_abstract  # noqa: PLC0415

        def forward(*inner_args: Any, **inner_kwargs: Any) -> _PrimitiveExecution:
            normalized = self._bind_call(
                inner_args,
                {**inner_kwargs, **static_arguments},
            )
            return self._dispatch_normalized(normalized)

        def abstract_forward(*inner_args: Any, **inner_kwargs: Any) -> Any:
            return call_primitive_abstract(
                self,
                inner_args,
                {**inner_kwargs, **static_arguments},
            )

        return trace_primitive_call(
            forward,
            abstract_function=abstract_forward,
            op_name=self.op_name,
            schema_version=self._definition.schema_version,
            recorder=recorder,
            args=(),
            kwargs=dynamic_arguments,
            node_attrs=static_arguments,
            nondiff_argnames=frozenset(self.nondiff_argnames),
            dynamic_argnames=frozenset(self._dynamic_argnames),
            has_residual=self.has_residual,
            track_output_arity=track_output_arity,
        )

    def _call_dynamic_only(self, **arguments: Any) -> Any:
        """Invoke a stable internal primitive whose Python state cannot be staged."""
        if not is_tracing() or _get_active_trace_kind() == "stage_abstract":
            msg = "A dynamic-only primitive requires an active concrete trace"
            raise RuntimeError(msg)
        normalized = self._bind_call((), arguments)
        return self._trace_dynamic_call(normalized, track_output_arity=False)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.has_residual and _is_rematerializing():
            msg = (
                f"Checkpointed regions cannot call residual primitive '{self.name}'. "
                "Move the primitive outside ad.checkpoint()."
            )
            raise TracingError(msg)
        arguments = self._bind_call(cast("tuple[Any, ...]", args), kwargs)

        if not is_tracing():
            return cast("R", self._dispatch_impl(**arguments))

        if _get_active_trace_kind() == "stage_abstract":
            if self._definition.variable_output_arity:
                msg = (
                    f"Primitive '{self.name}' has variable output arity and supports "
                    "concrete dynamic transforms only"
                )
                raise TracingError(msg)
            from advect.core._stage import call_primitive_abstract  # noqa: PLC0415

            static_arguments, dynamic_arguments = self._partition_arguments(arguments)
            return cast(
                "R",
                call_primitive_abstract(
                    self,
                    (),
                    {**dynamic_arguments, **static_arguments},
                ),
            )

        return cast(
            "R",
            self._trace_dynamic_call(
                arguments,
                track_output_arity=not self._definition.variable_output_arity,
            ),
        )

    def def_abstract(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Attach the primitive's abstract staging rule.

        The rule has the implementation's fixed named parameters. Advect
        preserves each dynamic argument's pytree while replacing its
        array/scalar leaves with ``advect.AbstractValue``; declared static
        arguments arrive unchanged. Return the concrete output pytree with
        ``advect.ArraySpec`` or ``AbstractValue`` leaves.

        The function is returned unchanged so this method can be used as a
        decorator.
        """
        if self._abstract_rule is not None:
            msg = f"Primitive '{self.name}' already has abstract evaluation"
            raise ValueError(msg)
        self._validate_abstract_signature(fn)
        get_registry().update(self._op_name, abstract_rule=fn)
        return fn

    def _rule_static_attrs(self, attrs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        meta, node_attrs = _split_primitive_attrs(attrs)
        return meta, {name: node_attrs[name] for name in self.static_argnames}

    def def_jvp(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Attach ``fn(output, primals, tangents, **static_attrs)`` as the JVP.

        ``output`` has the implementation's public output pytree. ``primals``
        and ``tangents`` are flat tuples with one entry per dynamic
        array/scalar leaf, in implementation-parameter and pytree order.
        Tangents may be ``None`` for inactive leaves and are always ``None``
        for leaves of a declared nondifferentiable argument. Static arguments
        are passed by name. Return a tangent with the output pytree.

        Write the rule as traceable, real-linear code so Advect can transpose
        it structurally and differentiate it again. The function is returned
        unchanged for decorator use.
        """
        if self._jvp_rule is not None:
            msg = f"Primitive '{self.name}' already has a JVP rule"
            raise ValueError(msg)
        self._validate_derivative_signature(fn, rule="JVP", positional_count=3)

        @functools.wraps(fn)
        def runtime_jvp(
            answer: Any,
            *inputs: Any,
            tangents: tuple[Any | None, ...],
            **attrs: Any,
        ) -> Any:
            meta, static_attrs = self._rule_static_attrs(attrs)
            nondiff_mask = meta.nondiff_mask(len(inputs))
            active_tangents = tuple(
                None if nondiff else tangent
                for nondiff, tangent in zip(nondiff_mask, tangents, strict=True)
            )
            public_output = _reconstruct_primitive_output(
                meta,
                answer,
                label="JVP primal output",
            )
            output_tangent = fn(
                public_output,
                tuple(inputs),
                active_tangents,
                **static_attrs,
            )
            return _flatten_primitive_output(
                meta,
                output_tangent,
                label="JVP result",
            )

        get_registry().register_jvp(self.op_name, runtime_jvp)
        return fn

    def def_transpose(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Attach an ordinary or exact-residual transpose rule.

        Ordinary primitives receive
        ``(cotangent, primals, output, **static_attrs)``.
        A primitive declared with ``residual=True`` receives
        ``(cotangent, primals, output, residual, **static_attrs)``.
        ``cotangent`` and ``output`` have the public output pytree; ``primals``
        is the same flattened dynamic-leaf tuple used by the JVP. Return a flat
        tuple with one contribution per dynamic leaf in that order. Advect
        suppresses contributions for declared nondifferentiable arguments.

        A rule may accept the optional keyword-only
        ``active_input_indices=None`` and return ``None`` for inactive
        contributions to avoid unnecessary work. Add an explicit transpose
        only when structural transposition cannot express the correct real
        adjoint, when an exact residual is required, or when measurement
        justifies a direct rule. The function is returned unchanged for
        decorator use.
        """
        if self._transpose_rule is not None:
            msg = f"Primitive '{self.name}' already has a transpose rule"
            raise ValueError(msg)
        positional_count = 4 if self.has_residual else 3
        self._validate_derivative_signature(
            fn,
            rule="transpose",
            positional_count=positional_count,
        )
        active_parameter = _callable_signature(
            fn,
            label=f"transpose rule for primitive {self.name!r}",
        ).parameters.get("active_input_indices")
        if (
            active_parameter is not None
            and active_parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        ):
            msg = f"Primitive '{self.name}' transpose active_input_indices must be keyword-only"
            raise TypeError(msg)

        def apply_rule(
            answer: Any,
            inputs: tuple[Any, ...],
            g: Any,
            residual: Any | None,
            attrs: Mapping[str, Any],
            *,
            active_input_indices: tuple[int, ...] | None = None,
        ) -> tuple[Any | None, ...]:
            meta, static_attrs = self._rule_static_attrs(attrs)
            public_output = _reconstruct_primitive_output(
                meta,
                answer,
                label="transpose primal output",
            )
            public_cotangent = _reconstruct_primitive_output(
                meta,
                g,
                label="transpose cotangent",
            )
            if active_parameter is not None:
                static_attrs["active_input_indices"] = active_input_indices
            if self.has_residual:
                result = fn(
                    public_cotangent,
                    inputs,
                    public_output,
                    residual,
                    **static_attrs,
                )
            else:
                result = fn(
                    public_cotangent,
                    inputs,
                    public_output,
                    **static_attrs,
                )
            contributions = _flatten_input_gradients(
                result,
                expected_input_count=len(inputs),
            )
            nondiff_mask = meta.nondiff_mask(len(inputs))
            return tuple(
                None if nondiff else contribution
                for nondiff, contribution in zip(
                    nondiff_mask,
                    contributions,
                    strict=True,
                )
            )

        @functools.wraps(fn)
        def runtime_transpose(
            answer: Any,
            *inputs: Any,
            g: Any,
            residual: Any | None = None,
            **attrs: Any,
        ) -> tuple[Any | None, ...]:
            return apply_rule(
                answer,
                tuple(inputs),
                g,
                residual,
                attrs,
            )

        if active_parameter is not None:

            @functools.wraps(fn)
            def selective_runtime_transpose(
                answer: Any,
                *inputs: Any,
                g: Any,
                active_input_indices: tuple[int, ...],
                residual: Any | None = None,
                **attrs: Any,
            ) -> tuple[Any | None, ...]:
                return apply_rule(
                    answer,
                    tuple(inputs),
                    g,
                    residual,
                    attrs,
                    active_input_indices=active_input_indices,
                )

            setattr(
                runtime_transpose,
                _SELECT_INPUTS_VJP_ATTR,
                selective_runtime_transpose,
            )

        get_registry().register_vjp(
            self.op_name,
            runtime_transpose,
            needs_inputs=True,
        )
        return fn


@overload
def primitive[**CallP, ResultT](
    function: Callable[CallP, ResultT],
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
    variable_output_arity: bool = False,
) -> Primitive[CallP, ResultT]: ...


@overload
def primitive[**CallP, ResultT](
    function: None = None,
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
    variable_output_arity: bool = False,
) -> Callable[[Callable[CallP, ResultT]], Primitive[CallP, ResultT]]: ...


def primitive[**CallP, ResultT](
    function: Callable[CallP, ResultT] | None = None,
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
    variable_output_arity: bool = False,
) -> Primitive[CallP, ResultT] | Callable[[Callable[CallP, ResultT]], Primitive[CallP, ResultT]]:
    """Define one atomic operation from its concrete implementation.

    The implementation must have fixed named parameters: positional-or-keyword
    and keyword-only parameters are supported, while positional-only
    parameters, ``*args``, and ``**kwargs`` are rejected. Calls still follow
    the implementation's normal Python signature.

    ``static_argnames`` removes complete named arguments from tracing and
    stores them as operation attributes. ``nondiff_argnames`` keeps complete
    arguments as dynamic operands but supplies ``None`` tangents and suppresses
    their transpose contributions. The two sets must be disjoint. Derivative
    rules receive all remaining dynamic array/scalar leaves flattened in
    implementation-parameter and pytree order.

    With ``residual=True``, the implementation must return
    ``advect.PrimitiveResult``; callers still receive only its ``output``.
    With ``variable_output_arity=True``, each concrete dynamic invocation owns
    its output leaf count; abstract staging remains unsupported. Rules are
    attached to the returned handle.

    Parameters
    ----------
    function
        Concrete implementation, when the decorator is applied directly.
    name
        Operation identity without the internal ``custom.`` prefix. By
        default Advect uses the implementation's module and qualified name.
        Use a stable explicit name for serialized artifacts.
    static_argnames
        Complete implementation arguments treated as concrete configuration.
    nondiff_argnames
        Complete dynamic arguments excluded from differentiation.
    residual
        Whether the implementation returns an invocation-local
        ``PrimitiveResult`` for an exact transpose.
    variable_output_arity
        Whether concrete invocations may return different numbers of output
        leaves. Such primitives cannot be staged.

    Returns
    -------
    Primitive or callable
        A callable authoring handle, or a decorator that creates one.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> @ad.primitive(name="examples.cube")
    ... def cube(value):
    ...     return value**3
    >>> @cube.def_abstract
    ... def cube_abstract(value):
    ...     return value.spec
    >>> @cube.def_jvp
    ... def cube_jvp(output, primals, tangents):
    ...     del output
    ...     (value,), (tangent,) = primals, tangents
    ...     return np.zeros_like(value) if tangent is None else 3 * value**2 * tangent
    >>> from advect.testing import check_primitive
    >>> sample = np.array([2.0])
    >>> check_primitive(
    ...     cube,
    ...     primals=(sample,),
    ...     check=("abstract", "jvp", "transpose", "nested", "stage"),
    ... )
    >>> ad.grad(lambda value: np.sum(cube(value)))(sample).tolist()
    [12.0]
    """

    def define(implementation: Callable[CallP, ResultT]) -> Primitive[CallP, ResultT]:
        operation_name = name
        if operation_name is None:
            operation_name = f"{implementation.__module__}.{implementation.__qualname__}"
        return Primitive(
            operation_name,
            implementation,
            static_argnames=static_argnames,
            nondiff_argnames=nondiff_argnames,
            residual=residual,
            variable_output_arity=variable_output_arity,
        )

    return define if function is None else define(function)


def evaluate_primitive(
    op: str,
    input_values: tuple[Any, ...],
    attrs: Mapping[str, Any],
    *,
    namespace: Any | None = None,
) -> Any:
    """Evaluate a linked primitive node with concrete values."""
    primitive = Primitive._linked(op)  # noqa: SLF001 - internal replay link
    meta, node_attrs = _split_primitive_attrs(attrs)
    args, kwargs = _reconstruct_primitive_call(meta, input_values)
    kwargs.update(node_attrs)
    if _get_active_trace_kind() == "autodiff_dynamic":
        # Preserve one atomic custom node when a durable program is replayed
        # under differentiation. Implementation execution happens inside the
        # primitive boundary, where residual ownership can transfer to the
        # enclosing dynamic tape.
        result = primitive(*args, **kwargs)
    else:
        normalized = primitive._bind_call(args, kwargs)  # noqa: SLF001
        with primitive._dispatch_normalized(normalized) as execution:  # noqa: SLF001
            result = execution.output
    leaves, treedef = _normalize_output_pytree(
        result,
        namespace=namespace or _infer_namespace(input_values),
    )
    _validate_output_treedef(meta, treedef, op=op)
    return leaves[0] if len(leaves) == 1 else tuple(leaves)


__all__ = [
    "MissingPrimitiveRuleError",
    "evaluate_primitive",
    "primitive",
]
