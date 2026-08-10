"""Reverse-mode transform API across dynamic calls and staged programs.

This module owns input selection and result contracts for `grad`,
`value_and_grad`, `vjp`, and `vjp_program`. Ordinary calls consume the
invocation-local tape and linear-map machinery; staged calls ask
`StagedProgram` to compile the same reverse transform into a durable graph.
Keeping lifetime dispatch here lets both paths share public semantics while
Python core retains the outer program envelope and `advect-runtime` retains
the graph format, validation, and scheduling.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Self, cast

from advect.autodiff._ephemeral import (
    LinearMap,
    apply_unary_array_pullback,
    linearize_call,
    trace_unary_array_call,
    unary_array_trace_provider,
)
from advect.autodiff.api._reverse_scalars import (
    _extract_scalar_output,
    _scalar_cotangent_for_output,
    _scalar_cotangent_leaf,
)
from advect.autodiff.api._scalar_boundary import _is_complex_numeric, _unlift_scalar_array
from advect.autodiff.api.inputs import (
    _get_positional_param_names,
    _get_signature,
    _normalize_argnums_for_call,
    _normalize_argnums_spec,
    _prefix_for_argnum,
    _validate_argnames,
)
from advect.core._abstract_model import ArraySpec
from advect.core._context import _get_active_trace_kind
from advect.core._protocols import _snapshot_traced
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.core._stage import StagedProgram

if TYPE_CHECKING:
    from collections.abc import Callable


_CACHE_MISS = object()


class Pullback:
    """One-shot reverse linearization returned by `vjp`.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> _, pullback = ad.vjp(lambda x: x**2)(np.array([1.0, 2.0]))
    >>> pullback(np.ones(2)).tolist()
    [2.0, 4.0]
    """

    def __init__(self, linear: LinearMap) -> None:
        self._linear = linear

    def __call__(self, cotangent: Any) -> Any:
        """Apply the pullback once and release its retained trace."""
        return self._linear._consume_pullback(cotangent)  # noqa: SLF001

    def close(self) -> None:
        """Release the retained trace without applying the pullback."""
        self._linear.close()

    def __enter__(self) -> Self:
        """Enter an ownership scope for the pending pullback."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release the pullback when leaving its ownership scope."""
        self.close()


class _UnaryArrayProviderCache:
    """Cache only type-stable input plumbing, never a traced program."""

    __slots__ = ("_input_type", "_result")

    def __init__(self) -> None:
        self._input_type: type[Any] | None = None
        self._result: tuple[bool, Any] | object = _CACHE_MISS

    def resolve(self, value: object) -> tuple[bool, Any]:
        value_type = type(value)
        if self._input_type is value_type and self._result is not _CACHE_MISS:
            return cast("tuple[bool, Any]", self._result)

        result = unary_array_trace_provider(value)
        attrs = getattr(value, "__dict__", None)
        instance_specific = (isinstance(attrs, dict) and "__array_namespace__" in attrs) or bool(
            getattr(value_type, "__advect_namespace_is_instance_specific__", False)
        )
        if not instance_specific:
            self._input_type = value_type
            self._result = result
        return result


def _selected_arguments(
    f: Callable[..., Any],
    *,
    argnums: int | tuple[int, ...] | None,
    argnames: tuple[str, ...] | None,
) -> tuple[tuple[int, ...], bool]:
    if argnames is not None and not isinstance(f, StagedProgram):
        _validate_argnames(_get_signature(f), argnames)
    resolved = 0 if argnums is None and argnames is None else (() if argnums is None else argnums)
    return _normalize_argnums_spec(resolved)


def _staged_gradient_scalar_mask(
    f: StagedProgram,
    *,
    argnums: tuple[int, ...],
    single_argnum: bool,
    argnames: tuple[str, ...] | None,
) -> tuple[bool, ...]:
    """Return the weak-scalar category of each selected staged input leaf."""
    positional_specs, named_specs = f.signature
    selected_argnums = _normalize_argnums_for_call(argnums, nargs=len(positional_specs))
    positional = [positional_specs[index] for index in selected_argnums]
    names = () if argnames is None else argnames
    missing = [name for name in names if name not in named_specs]
    if missing:
        msg = f"Staged argument name(s) {missing!r} are not present in the compiled signature"
        raise ValueError(msg)
    named = {name: named_specs[name] for name in names}

    selected_leaves, _selected_treedef = tree_flatten((tuple(positional), named))
    for spec in selected_leaves:
        if not isinstance(spec, ArraySpec):
            continue
        dtype = str(spec.dtype).lower()
        if spec.weak and not dtype.startswith("float"):
            msg = (
                "Differentiating a staged weak scalar requires a real floating signature; "
                f"got dtype={spec.dtype!r}. Use a strong rank-zero array for complex "
                "differentiation."
            )
            raise TypeError(msg)

    positional_masks = [
        tree_unflatten(
            treedef,
            [isinstance(spec, ArraySpec) and spec.weak and spec.shape == () for spec in leaves],
        )
        for spec_tree in positional
        for leaves, treedef in [tree_flatten(spec_tree)]
    ]
    named_masks = {
        name: tree_unflatten(
            treedef,
            [isinstance(spec, ArraySpec) and spec.weak and spec.shape == () for spec in leaves],
        )
        for name, spec_tree in named.items()
        for leaves, treedef in [tree_flatten(spec_tree)]
    }
    if named_masks:
        if not positional_masks:
            result: Any = named_masks
        else:
            positional_result: Any = (
                positional_masks[0]
                if single_argnum and len(positional_masks) == 1
                else tuple(positional_masks)
            )
            result = positional_result, named_masks
    elif single_argnum:
        result = positional_masks[0]
    else:
        result = tuple(positional_masks)
    leaves, _treedef = tree_flatten(result)
    return tuple(bool(leaf) for leaf in leaves)


def _reject_complex_grad_output(out_leaf: Any) -> None:
    if _is_complex_numeric(out_leaf):
        msg = (
            "grad requires a real scalar output. For complex outputs use "
            "linearize(), jvp(), or vjp(); Advect does not guess a holomorphic convention."
        )
        raise ValueError(msg)


def _materialize_aux(aux: Any) -> Any:
    """Snapshot auxiliary tracer leaves while their trace is active."""
    leaves, treedef = tree_flatten(aux)
    concrete: list[Any] = []
    for leaf in leaves:
        restore_python_scalar = bool(getattr(leaf, "_advect_weak", False))
        value = leaf
        while callable(getattr(value, "_advect_snapshot", None)):
            _node_id, next_value = _snapshot_traced(value)
            if next_value is value:
                break
            value = next_value
        concrete.append(_unlift_scalar_array(value) if restore_python_scalar else value)
    return tree_unflatten(treedef, concrete)


def _unary_array_fast_path(
    f: Callable[..., Any],
    *,
    argnums: tuple[int, ...],
    single_argnum: bool,
    argnames: tuple[str, ...] | None,
    has_aux: bool,
) -> tuple[bool, str]:
    enabled = single_argnum and argnums == (0,) and argnames is None and not has_aux
    if not enabled:
        return False, "arg0"
    fixed_names, varargs_name = _get_positional_param_names(f)
    return True, _prefix_for_argnum(0, fixed=fixed_names, varargs_name=varargs_name)


def _try_unary_array_value_and_grad(
    f: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_name: str,
    provider_cache: _UnaryArrayProviderCache,
    transform_name: str,
) -> tuple[Any, Any] | None:
    # The concrete unary fast path performs provider introspection that an
    # enclosing abstract trace deliberately cannot answer. The general
    # dynamic path can use that outer tracer as its provider value and emit
    # the derivative program into the enclosing staged graph.
    if _get_active_trace_kind() == "stage_abstract":
        return None
    if not args:
        return None
    supported, provider = provider_cache.resolve(args[0])
    if not supported:
        return None
    trace = trace_unary_array_call(
        f,
        args=args,
        kwargs=kwargs,
        input_name=input_name,
        provider=provider,
    )
    try:
        cotangent = _real_scalar_seed(
            trace.output,
            output_is_leaf=trace.output_treedef.node_type is None,
            transform_name=transform_name,
        )
        return trace.output, apply_unary_array_pullback(trace, cotangent)
    except Exception:
        trace.tape.release_payloads()
        raise


def _real_scalar_seed(
    output: Any,
    *,
    output_is_leaf: bool,
    transform_name: str,
) -> Any:
    """Validate and seed the common concrete rank-zero output directly."""
    if output_is_leaf and getattr(output, "shape", None) == ():
        dtype = getattr(output, "dtype", None)
        kind = getattr(dtype, "kind", None)
        if kind == "c":
            _reject_complex_grad_output(output)
        scalar_type = getattr(dtype, "type", None)
        output_provider = type(output).__module__.partition(".")[0]
        scalar_provider = str(getattr(scalar_type, "__module__", "")).partition(".")[0]
        if callable(scalar_type) and output_provider == scalar_provider:
            return scalar_type(1)
        return _scalar_cotangent_leaf(output)

    out_leaf, _out_treedef = _extract_scalar_output(output, transform_name=transform_name)
    _reject_complex_grad_output(out_leaf)
    return _scalar_cotangent_leaf(out_leaf)


def vjp(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Pullback]]:
    """Return a concrete value and a one-shot reverse pullback.

    `vjp` is always a dynamic transform. Each call traces the selected concrete
    inputs and returns a `Pullback` that owns that invocation's tape, retained
    provider values, and primitive residuals. This remains true when `f` is a
    `StagedProgram`; use `vjp_program` to compile a reusable staged pullback.

    Parameters
    ----------
    f
        Callable to linearize. Its output may be any supported array or pytree.
    argnums
        Positional arguments to differentiate. An integer makes the pullback
        return that argument's gradient pytree directly; a tuple makes it
        return a tuple in the given order. Negative indices are resolved for
        each call.

    Returns
    -------
    Callable
        A concrete-tracing function returning `(value, pullback)`. `value`
        preserves the callable's output pytree. Call `pullback(cotangent)` with
        a cotangent matching that pytree to obtain the selected input
        gradients.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the transformed call.
    TypeError
        If a selected input is an unsupported Python complex scalar, or a
        cotangent leaf has an invalid numeric category.
    ValueError
        If positional selections are duplicated or the cotangent pytree or
        leaf shape does not match the output.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.
    RuntimeError
        If the pullback is applied after it has already been consumed or
        closed.

    Notes
    -----
    Applying the pullback consumes it and releases its retained trace. Call
    `close()` to release it without applying it, or use it as a context manager
    when deterministic cleanup matters.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> value, pullback = ad.vjp(lambda x: x**2)(np.array([1.0, 2.0]))
    >>> value.tolist()
    [1.0, 4.0]
    >>> pullback(np.ones(2)).tolist()
    [2.0, 4.0]
    """
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)

    @functools.wraps(f)
    def vjp_fn(*args: Any, **kwargs: Any) -> tuple[Any, Pullback]:
        value, linear = linearize_call(
            f,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=None,
            single_argnum=single_argnum,
            reverse_only=True,
        )

        pullback = Pullback(linear)
        return linear._unlift_outputs(value), pullback  # noqa: SLF001

    return vjp_fn


def vjp_program(
    f: StagedProgram,
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
) -> StagedProgram:
    """Compile a reusable staged pullback program.

    Parameters
    ----------
    f
        Primal `StagedProgram` to transpose. Ordinary callables are not
        accepted; use `vjp` for a concrete dynamic pullback.
    argnums
        Positional inputs to differentiate. An integer returns that input's
        gradient pytree directly, while a tuple returns a tuple in the given
        order. `None` selects input zero unless `argnames` is provided, in
        which case it selects no positional inputs. Negative indices are
        resolved against the program's positional signature.
    argnames
        Keyword inputs from the staged signature to differentiate. Their
        gradients are returned in a dictionary keyed by name. When positional
        and named inputs are both selected, the result is
        `(positional_gradients, named_gradients)`; the positional part follows
        the integer-versus-tuple rule above.

    Returns
    -------
    StagedProgram
        An immutable, serializable program with the primal call signature plus
        a reserved keyword-only `cotangent` input. The cotangent has the
        primal output's pytree and leaf specifications. The program preserves
        the primal program's Array API revision.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the staged signature.
    TypeError
        If `f` is not a `StagedProgram`, or a selected weak scalar signature is
        not real floating-point.
    ValueError
        If positional selections are duplicated, a selected input is absent
        from the staged signature, or the primal signature already reserves
        the `cotangent` keyword.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> square = ad.stage(lambda x: x**2, np.array([1.0, 2.0]))
    >>> pullback = ad.vjp_program(square)
    >>> pullback(np.array([3.0, 4.0]), cotangent=np.ones(2)).tolist()
    [6.0, 8.0]
    """
    if not isinstance(f, StagedProgram):
        msg = "vjp_program() requires a StagedProgram; call stage() first"
        raise TypeError(msg)

    argnums_tuple, single_argnum = _selected_arguments(
        f,
        argnums=argnums,
        argnames=argnames,
    )
    scalar_mask = _staged_gradient_scalar_mask(
        f,
        argnums=argnums_tuple,
        single_argnum=single_argnum,
        argnames=argnames,
    )

    def pullback_program(*args: Any, cotangent: Any, **kwargs: Any) -> Any:
        _value, linear = linearize_call(
            f,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=argnames,
            single_argnum=single_argnum,
            reverse_only=True,
        )
        try:
            return linear._consume_pullback(cotangent)  # noqa: SLF001
        except Exception:
            linear.close()
            raise

    return f._staged_transform(  # noqa: SLF001
        pullback_program,
        output_argname="cotangent",
        scalar_output_override=(0, scalar_mask),
    )


def _dynamic_grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Project the gradient from the shared concrete reverse transform."""
    transformed = _dynamic_value_and_grad(
        f,
        argnums=argnums,
        argnames=argnames,
        has_aux=has_aux,
        transform_name="grad",
    )

    def grad_fn(*args: Any, **kwargs: Any) -> Any:
        result = transformed(*args, **kwargs)
        if has_aux:
            _value, gradients, aux = result
            return gradients, aux
        _value, gradients = result
        return gradients

    if not isinstance(f, StagedProgram):
        functools.update_wrapper(grad_fn, f)
    return grad_fn


def _dynamic_value_and_grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
    transform_name: str = "value_and_grad",
) -> Callable[..., tuple[Any, ...]]:
    """Build the concrete value-and-gradient transform for either lifetime."""
    argnums_tuple, single_argnum = _selected_arguments(
        f,
        argnums=argnums,
        argnames=argnames,
    )
    use_unary_fast_path, input_name = _unary_array_fast_path(
        f,
        argnums=argnums_tuple,
        single_argnum=single_argnum,
        argnames=argnames,
        has_aux=has_aux,
    )
    provider_cache = _UnaryArrayProviderCache()

    def value_and_grad_fn(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if use_unary_fast_path:
            fast_result = _try_unary_array_value_and_grad(
                f,
                args=args,
                kwargs=kwargs,
                input_name=input_name,
                provider_cache=provider_cache,
                transform_name=transform_name,
            )
            if fast_result is not None:
                return fast_result

        aux_box: list[Any] = []
        trace_target = f
        if has_aux:

            def trace_target(*inner_args: Any, **inner_kwargs: Any) -> Any:
                value, aux = f(*inner_args, **inner_kwargs)
                aux_box.append(_materialize_aux(aux))
                return value

        value, linear = linearize_call(
            trace_target,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=argnames,
            single_argnum=single_argnum,
            reverse_only=True,
        )
        try:
            out_leaf, out_treedef = _extract_scalar_output(
                value,
                transform_name=transform_name,
            )
            _reject_complex_grad_output(out_leaf)
            cotangent = _scalar_cotangent_for_output(
                out_leaf=out_leaf,
                out_treedef=out_treedef,
            )
            gradients = linear._consume_pullback(cotangent)  # noqa: SLF001
        except Exception:
            linear.close()
            raise
        if has_aux:
            return (
                linear._unlift_outputs(value),  # noqa: SLF001
                gradients,
                _materialize_aux(aux_box[-1]),
            )
        return linear._unlift_outputs(value), gradients  # noqa: SLF001

    if not isinstance(f, StagedProgram):
        functools.update_wrapper(value_and_grad_fn, f)
    return value_and_grad_fn


def grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Differentiate a scalar-valued function with reverse mode.

    An ordinary callable is traced from concrete inputs on every invocation;
    no trace or graph is cached between calls. Passing a `StagedProgram`
    instead compiles and returns another immutable staged program. Warm calls
    to that program execute its prebuilt graph without a dynamic tape or
    reverse sweep.

    Parameters
    ----------
    f
        Callable whose differentiated value is a real scalar or a one-leaf
        pytree containing a real scalar. With `has_aux=True`, it instead
        returns `(value, auxiliary)`, where only `value` is differentiated.
        A `StagedProgram` is also accepted.
    argnums
        Positional arguments to differentiate. An integer returns that
        argument's gradient pytree directly, while a tuple returns a tuple in
        the given order. `None` selects argument zero unless `argnames` is
        provided, in which case it selects no positional arguments. Negative
        indices are resolved for each call.
    argnames
        Named arguments to differentiate. Their gradients are returned in a
        dictionary keyed by name. For an ordinary callable, a selected name
        may be passed positionally or by keyword; staged named inputs must be
        passed by keyword. When positional and named inputs are both selected,
        the result is `(positional_gradients, named_gradients)`, with the
        positional part following the integer-versus-tuple rule above.
    has_aux
        Whether `f` returns `(value, auxiliary)`. The auxiliary value is
        excluded from differentiation. A dynamic call materializes it as a
        concrete sidecar; a staged transform records it as an ordinary staged
        output.

    Returns
    -------
    Callable or StagedProgram
        For an ordinary callable, a concrete-tracing gradient function. For a
        staged input, an immutable, serializable derivative program with the
        same input signature and Array API revision. Its result has the
        gradient structure selected by `argnums` and `argnames`; when
        `has_aux=True`, it returns `(gradient, auxiliary)`.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the transformed call.
    TypeError
        If a selected input is an unsupported Python complex scalar, or a
        selected staged weak-scalar signature is not real floating-point.
    ValueError
        If positional selections are duplicated, an ordinary callable
        argument is selected both positionally and by name, a selected name is
        unavailable, or the differentiated output is not a real scalar.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> ad.grad(lambda value: np.sum(value**2))(x).tolist()
    [2.0, 4.0, 6.0]
    """
    transformed = _dynamic_grad(
        f,
        argnums=argnums,
        argnames=argnames,
        has_aux=has_aux,
    )
    if isinstance(f, StagedProgram):
        argnums_tuple, single_argnum = _selected_arguments(
            f,
            argnums=argnums,
            argnames=argnames,
        )
        scalar_mask = _staged_gradient_scalar_mask(
            f,
            argnums=argnums_tuple,
            single_argnum=single_argnum,
            argnames=argnames,
        )
        return f._staged_transform(  # noqa: SLF001
            transformed,
            scalar_output_override=(0, scalar_mask),
        )
    return transformed


def value_and_grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[..., tuple[Any, ...]]:
    """Compute a scalar value and its reverse-mode gradient together.

    An ordinary callable is traced from concrete inputs on every invocation;
    no trace or graph is cached between calls. Passing a `StagedProgram`
    instead compiles and returns another immutable staged program. Warm calls
    to that program execute its prebuilt graph without a dynamic tape or
    reverse sweep.

    Parameters
    ----------
    f
        Callable whose differentiated value is a real scalar or a one-leaf
        pytree containing a real scalar. With `has_aux=True`, it instead
        returns `(value, auxiliary)`, where only `value` is differentiated.
        A `StagedProgram` is also accepted.
    argnums
        Positional arguments to differentiate. An integer returns that
        argument's gradient pytree directly, while a tuple returns a tuple in
        the given order. `None` selects argument zero unless `argnames` is
        provided, in which case it selects no positional arguments. Negative
        indices are resolved for each call.
    argnames
        Named arguments to differentiate. Their gradients are returned in a
        dictionary keyed by name. For an ordinary callable, a selected name
        may be passed positionally or by keyword; staged named inputs must be
        passed by keyword. When positional and named inputs are both selected,
        the gradient is `(positional_gradients, named_gradients)`, with the
        positional part following the integer-versus-tuple rule above.
    has_aux
        Whether `f` returns `(value, auxiliary)`. The auxiliary value is
        excluded from differentiation. A dynamic call materializes it as a
        concrete sidecar; a staged transform records it as an ordinary staged
        output.

    Returns
    -------
    Callable or StagedProgram
        For an ordinary callable, a concrete-tracing function returning
        `(value, gradient)`. For a staged input, an immutable, serializable
        program with the same input signature and Array API revision returning
        the same structure. With `has_aux=True`, either lifetime returns
        `(value, gradient, auxiliary)`.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the transformed call.
    TypeError
        If a selected input is an unsupported Python complex scalar, or a
        selected staged weak-scalar signature is not real floating-point.
    ValueError
        If positional selections are duplicated, an ordinary callable
        argument is selected both positionally and by name, a selected name is
        unavailable, or the differentiated output is not a real scalar.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> value, gradient = ad.value_and_grad(lambda v: np.sum(v**2))(x)
    >>> float(value), gradient.tolist()
    (14.0, [2.0, 4.0, 6.0])
    """
    transformed = _dynamic_value_and_grad(
        f,
        argnums=argnums,
        argnames=argnames,
        has_aux=has_aux,
    )
    if isinstance(f, StagedProgram):
        argnums_tuple, single_argnum = _selected_arguments(
            f,
            argnums=argnums,
            argnames=argnames,
        )
        scalar_mask = _staged_gradient_scalar_mask(
            f,
            argnums=argnums_tuple,
            single_argnum=single_argnum,
            argnames=argnames,
        )
        return f._staged_transform(  # noqa: SLF001
            transformed,
            scalar_output_override=(1, scalar_mask),
        )
    return transformed


__all__ = ["LinearMap", "Pullback", "grad", "value_and_grad", "vjp", "vjp_program"]
