"""Concrete dynamic linearization and forward-mode transforms."""

from __future__ import annotations

import functools
import math
from itertools import batched
from typing import TYPE_CHECKING, Any

from advect.autodiff._ephemeral import (
    _PULLBACK_MANY_BATCH_SIZE,
    LinearMap,
    linearize_call,
)
from advect.autodiff.api._pullback_values import (
    _format_backward_result,
    _zeros_like,
)
from advect.autodiff.api._scalar_boundary import _unlift_scalar_array
from advect.autodiff.api.inputs import _normalize_argnums_spec
from advect.core._array_api.providers import _get_array_namespace
from advect.core._context import _use_array_api_version
from advect.core._pytree import tree_flatten, tree_unflatten
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable


def linearize(
    f: Callable[..., Any],
    *primals: Any,
    argnums: int | tuple[int, ...] = 0,
    **kwargs: Any,
) -> tuple[Any, LinearMap]:
    """Linearize one concrete call and return its reusable real-linear map.

    The call is traced immediately from ``primals`` and ``kwargs``. The
    returned ``LinearMap`` owns that invocation's tape, retained provider
    values, and primitive residuals; it is not a cached or durable program.

    Parameters
    ----------
    f
        Callable to linearize. Its output may be any supported array or
        pytree. A ``StagedProgram`` is accepted, but the surrounding
        linearization is still concrete and invocation-local.
    *primals
        Positional arguments for this call. Only the arguments selected by
        ``argnums`` are tangent inputs; the others remain primal
        coefficients.
    argnums
        Positional arguments to differentiate. An integer makes
        ``linear(tangents)`` accept that argument's tangent pytree directly; a
        tuple makes it accept a tuple of tangent pytrees in the given order.
        Negative indices are resolved against ``primals``.
    **kwargs
        Keyword arguments forwarded to ``f``. ``linearize`` does not select
        keyword arguments for differentiation.

    Returns
    -------
    value
        The concrete output of ``f``, with its pytree structure preserved.
    linear
        A reusable ``LinearMap``. Calling ``linear(tangents)`` applies the JVP
        and returns a tangent with the output pytree. Calling
        ``linear.pullback(cotangent)`` or ``linear.transpose()(cotangent)``
        applies the real adjoint and returns the selected input structure.

    Raises
    ------
    IndexError
        If a positional selection is out of range.
    TypeError
        If a selected input contains an unsupported Python complex scalar, or
        a tangent has an invalid structure or numeric category.
    ValueError
        If positional selections are duplicated, or a tangent pytree or leaf
        shape does not match its selected primal.
    NoJVPError
        If the returned map is applied forward through an operation without a
        JVP rule.
    NoVJPError
        If the returned map is transposed through an operation without an
        explicit or structurally derivable transpose rule.
    RuntimeError
        If the map is applied after it has been closed.

    Notes
    -----
    The map remains reusable until ``close()``. Close it explicitly, or use it
    as a context manager, to release retained concrete values and residuals
    deterministically.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> value, linear = ad.linearize(lambda x: x**2, np.array([1.0, 2.0]))
    >>> value.tolist()
    [1.0, 4.0]
    >>> with linear:
    ...     linear(np.ones(2)).tolist()
    [2.0, 4.0]
    """
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)
    value, linear = linearize_call(
        f,
        args=primals,
        kwargs=kwargs,
        argnums=argnums_tuple,
        argnames=None,
        single_argnum=single_argnum,
    )
    return linear._unlift_outputs(value), linear  # noqa: SLF001


def jvp(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Any]]:
    """Return a concrete-tracing Jacobian-vector product transform.

    Parameters
    ----------
    f
        Callable to differentiate. Its output may be any supported array or
        pytree. Passing a ``StagedProgram`` executes it inside a concrete
        trace; this transform does not compile a new staged program.
    argnums
        Positional arguments to differentiate. An integer expects one tangent
        pytree directly. A tuple expects a tuple of tangent pytrees in the
        given order, including for a one-element tuple. Negative indices are
        resolved for each transformed call.

    Returns
    -------
    Callable
        A function called as ``transformed(*args, tangents=..., **kwargs)``.
        ``tangents`` must match the selected primal pytree or pytrees. The
        result is ``(value, output_tangent)``; both entries preserve ``f``'s
        output pytree, and disconnected output leaves receive zero tangents.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the transformed call.
    TypeError
        If a selected input is an unsupported Python complex scalar, multiple
        selected arguments are not given a tuple of tangents, or a tangent is
        supplied for a static or untraceable leaf.
    ValueError
        If positional selections are duplicated, or the tangent arity,
        pytree, or leaf shape does not match the selected primals.
    NoJVPError
        If an operation on the differentiated path has no JVP rule.

    Notes
    -----
    Each invocation traces the concrete values once, applies the JVP once, and
    releases the temporary ``LinearMap`` and any retained primitive residuals
    before returning. No trace is cached between calls.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> value, tangent = ad.jvp(lambda x: x**2)(np.array([1.0, 2.0]), tangents=np.ones(2))
    >>> value.tolist(), tangent.tolist()
    ([1.0, 4.0], [2.0, 4.0])
    """
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)

    @functools.wraps(f)
    def jvp_fn(*args: Any, tangents: Any, **kwargs: Any) -> tuple[Any, Any]:
        value, linear = linearize_call(
            f,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=None,
            single_argnum=single_argnum,
        )
        tangent = linear._consume(tangents)  # noqa: SLF001
        return linear._unlift_outputs(value), linear._unlift_outputs(tangent)  # noqa: SLF001

    return jvp_fn


def _is_complex_array(value: Any) -> bool:
    if isinstance(value, complex):
        return True
    dtype = getattr(value, "dtype", None)
    return bool(getattr(dtype, "kind", None) == "c" or "complex" in str(dtype).lower())


def _jacobian_selection(
    argnums: int | tuple[int, ...] | None,
    argnames: tuple[str, ...] | None,
) -> tuple[tuple[int, ...], bool]:
    resolved = 0 if argnums is None and argnames is None else (() if argnums is None else argnums)
    return _normalize_argnums_spec(resolved)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in getattr(value, "shape", ()))


def _basis_cotangent(value: Any, index: int) -> Any:
    shape = _shape(value)
    namespace = _get_array_namespace(value)
    if namespace is None:
        if shape:
            msg = "jacobian() output leaves require an array backend"
            raise RuntimeError(msg)
        return 1.0
    if shape == ():
        return namespace.ones_like(value)
    basis = namespace.zeros_like(value)
    flattened = namespace.reshape(basis, (-1,))
    flattened[index] = 1
    return namespace.reshape(flattened, shape)


def _basis_tangent(value: Any, index: int) -> Any:
    return _basis_cotangent(value, index)


def _basis_cotangent_rows(value: Any) -> Any:
    """Allocate every standard-basis cotangent as rows of one provider array."""
    shape = _shape(value)
    size = _flat_size(value)
    namespace = _get_array_namespace(value)
    if namespace is None:
        if shape:
            msg = "jacobian() output leaves require an array backend"
            raise RuntimeError(msg)
        return (1.0,)

    kwargs: dict[str, Any] = {"dtype": value.dtype}
    device = getattr(value, "device", None)
    if device is not None:
        kwargs["device"] = device
    identity = namespace.eye(size, **kwargs)
    return namespace.reshape(identity, (size, *shape))


def _basis_cotangent_row(rows: Any, index: int) -> Any:
    if isinstance(rows, tuple):
        return rows[index]
    return rows[(index, ...)]


def _flat_size(value: Any) -> int:
    shape = _shape(value)
    return math.prod(shape) if shape else 1


def _stack_jacobian_rows(
    rows: list[Any],
    *,
    output_shape: tuple[int, ...],
    output_like: Any,
) -> Any:
    if not rows:
        msg = "jacobian row stack cannot be empty"
        raise ValueError(msg)
    if all(row is None for row in rows):
        return None
    if any(row is None for row in rows):
        msg = "jacobian pullback changed disconnected input structure between seeds"
        raise RuntimeError(msg)

    sample = rows[0]
    sample_namespace = _get_array_namespace(sample)
    namespace = sample_namespace or _get_array_namespace(output_like)
    if namespace is None:
        if output_shape == () and len(rows) == 1 and _shape(sample) == ():
            return sample
        msg = "jacobian() requires an array backend to assemble dense derivative blocks"
        raise RuntimeError(msg)
    normalized = rows if sample_namespace is not None else [namespace.asarray(row) for row in rows]
    stacked = namespace.stack(tuple(normalized), axis=0)
    return namespace.reshape(stacked, (*output_shape, *_shape(sample)))


def _empty_jacobian_block(
    zero_gradient: Any,
    *,
    output_shape: tuple[int, ...],
    output_like: Any,
) -> Any:
    if zero_gradient is None:
        return None
    namespace = _get_array_namespace(zero_gradient) or _get_array_namespace(output_like)
    if namespace is None:
        msg = "jacobian() requires an array backend to assemble dense derivative blocks"
        raise RuntimeError(msg)
    zero = (
        zero_gradient
        if _get_array_namespace(zero_gradient) is not None
        else namespace.asarray(zero_gradient)
    )
    expanded = zero
    for _dimension in output_shape:
        expanded = namespace.expand_dims(expanded, axis=0)
    return namespace.broadcast_to(expanded, (*output_shape, *_shape(zero_gradient)))


def _validate_real_jacobian(linear: LinearMap, output_leaves: list[Any]) -> None:
    if any(_is_complex_array(leaf) for leaf in output_leaves):
        msg = "jacobian requires real outputs; use linearize() for complex real-linear maps"
        raise ValueError(msg)
    selected_specs = (
        *linear._trace.positional_specs,  # noqa: SLF001 - transform owns this map
        *linear._trace.named_specs.values(),  # noqa: SLF001 - transform owns this map
    )
    if any(
        leaf_spec.primal is not None and _is_complex_array(leaf_spec.primal)
        for spec in selected_specs
        for leaf_spec in spec.leaf_specs
    ):
        msg = "jacobian requires real inputs; use linearize() for complex real-linear maps"
        raise ValueError(msg)


def _input_gradient_leaf(leaf_spec: Any, gradients: dict[int, Any]) -> Any:
    node_id = leaf_spec.node_id
    if node_id is None:
        return None
    gradient = gradients.get(node_id)
    if gradient is None:
        primal = leaf_spec.primal
        gradient = _zeros_like(primal)
    return _unlift_scalar_array(gradient) if leaf_spec.restore_python_scalar else gradient


def _zero_input_gradient_leaf(leaf_spec: Any) -> Any:
    return _input_gradient_leaf(leaf_spec, {})


def _jacobian_reverse(
    linear: LinearMap,
    *,
    output_leaves: list[Any],
    output_treedef: Any,
) -> Any:
    if not output_leaves:
        return tree_unflatten(output_treedef, [])

    positional_specs, named_specs = _selected_input_specs(linear)
    leaf_specs = [
        leaf_spec
        for spec in (*positional_specs, *(spec for _name, spec in named_specs))
        for leaf_spec in spec.leaf_specs
    ]
    rows_by_output: list[list[list[Any]]] = [
        [[] for _leaf_spec in leaf_specs] for _output_leaf in output_leaves
    ]
    basis_rows = tuple(_basis_cotangent_rows(output_leaf) for output_leaf in output_leaves)
    seed_entries = (
        (
            output_index,
            {
                linear._trace.output_ids[output_index]: _basis_cotangent_row(  # noqa: SLF001
                    output_basis_rows,
                    basis_index,
                )
            },
        )
        for output_index, (output_leaf, output_basis_rows) in enumerate(
            zip(output_leaves, basis_rows, strict=True)
        )
        for basis_index in range(_flat_size(output_leaf))
    )
    for seed_batch in batched(seed_entries, _PULLBACK_MANY_BATCH_SIZE):
        gradient_sets = linear._transpose_seed_tables_many(  # noqa: SLF001
            tuple(seed_table for _output_index, seed_table in seed_batch)
        )
        for (output_index, _seed_table), gradients in zip(
            seed_batch,
            gradient_sets,
            strict=True,
        ):
            for rows, leaf_spec in zip(
                rows_by_output[output_index],
                leaf_specs,
                strict=True,
            ):
                rows.append(_input_gradient_leaf(leaf_spec, gradients))

    has_empty_output = any(_flat_size(output_leaf) == 0 for output_leaf in output_leaves)
    zero_gradient_leaves: list[Any] = (
        [_zero_input_gradient_leaf(leaf_spec) for leaf_spec in leaf_specs]
        if has_empty_output
        else []
    )
    output_blocks = []
    for output_leaf, rows_by_input in zip(
        output_leaves,
        rows_by_output,
        strict=True,
    ):
        output_shape = _shape(output_leaf)
        if _flat_size(output_leaf) == 0:
            blocks = [
                _empty_jacobian_block(
                    zero_gradient,
                    output_shape=output_shape,
                    output_like=output_leaf,
                )
                for zero_gradient in zero_gradient_leaves
            ]
        else:
            blocks = [
                _stack_jacobian_rows(
                    rows,
                    output_shape=output_shape,
                    output_like=output_leaf,
                )
                for rows in rows_by_input
            ]
        blocks = [
            _unlift_scalar_array(block) if leaf_spec.restore_python_scalar else block
            for block, leaf_spec in zip(blocks, leaf_specs, strict=True)
        ]
        output_blocks.append(_format_forward_input_blocks(linear, blocks))
    return tree_unflatten(output_treedef, output_blocks)


def _selected_input_specs(linear: LinearMap) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]:
    positional = tuple(linear._trace.positional_specs)  # noqa: SLF001
    named = tuple(linear._trace.named_specs.items())  # noqa: SLF001
    return positional, named


def _selected_input_size(linear: LinearMap) -> int:
    positional, named = _selected_input_specs(linear)
    return sum(
        _flat_size(leaf_spec.primal)
        for spec in (*positional, *(spec for _name, spec in named))
        for leaf_spec in spec.leaf_specs
        if leaf_spec.node_id is not None
    )


def _trace_uses_structural_transpose(linear: LinearMap) -> bool:
    registry = get_registry()
    return any(
        registry.has_jvp(op) and not registry.has_vjp(op)
        for op in linear._trace.tape.op_names  # noqa: SLF001 - transform owns this map
    )


def _trace_requires_reverse(linear: LinearMap) -> bool:
    """Return whether the trace contains an explicit transpose without a JVP."""
    registry = get_registry()
    return any(
        registry.has_vjp(op) and not registry.has_jvp(op)
        for op in linear._trace.tape.op_names  # noqa: SLF001 - transform owns this map
    )


def _stack_jacobian_columns(
    columns: list[Any],
    *,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    output_like: Any,
    input_like: Any,
) -> Any:
    expected = math.prod(input_shape) if input_shape else 1
    if len(columns) != expected:
        msg = f"jacobian JVP produced {len(columns)} columns for an input of size {expected}"
        raise RuntimeError(msg)
    if input_shape == ():
        return _cast_jacobian_block_like_input(columns[0], input_like)

    sample = columns[0]
    namespace = (
        _get_array_namespace(sample)
        or _get_array_namespace(output_like)
        or _get_array_namespace(input_like)
    )
    if namespace is None:
        msg = "jacobian() requires an array backend to assemble dense derivative blocks"
        raise RuntimeError(msg)
    normalized = [
        column if _get_array_namespace(column) is not None else namespace.asarray(column)
        for column in columns
    ]
    stacked = namespace.stack(tuple(normalized), axis=-1)
    block = namespace.reshape(stacked, (*output_shape, *input_shape))
    return _cast_jacobian_block_like_input(block, input_like)


def _cast_jacobian_block_like_input(block: Any, input_like: Any) -> Any:
    """Represent each dense block in its input tangent-space dtype."""
    input_dtype = getattr(input_like, "dtype", None)
    if input_dtype is None or getattr(block, "dtype", None) == input_dtype:
        return block
    namespace = _get_array_namespace(block) or _get_array_namespace(input_like)
    if namespace is not None and hasattr(namespace, "astype"):
        return namespace.astype(block, input_dtype, copy=False)
    astype = getattr(block, "astype", None)
    if callable(astype):
        return astype(input_dtype, copy=False)
    msg = "jacobian() cannot project a forward-mode block into the input dtype"
    raise RuntimeError(msg)


def _format_forward_input_blocks(
    linear: LinearMap,
    flat_blocks: list[Any],
) -> Any:
    positional_specs, named_specs = _selected_input_specs(linear)
    offset = 0
    positional = []
    for spec in positional_specs:
        end = offset + len(spec.leaf_specs)
        positional.append(tree_unflatten(spec.treedef, flat_blocks[offset:end]))
        offset = end
    named = {}
    for name, spec in named_specs:
        end = offset + len(spec.leaf_specs)
        named[name] = tree_unflatten(spec.treedef, flat_blocks[offset:end])
        offset = end
    if offset != len(flat_blocks):
        msg = "jacobian input block assembly did not consume every selected input leaf"
        raise RuntimeError(msg)
    return _format_backward_result(
        positional_grads=positional,
        named_grads=named,
        single_argnum=linear._single_argnum,  # noqa: SLF001
    )


def _jacobian_forward(
    linear: LinearMap,
    *,
    output_leaves: list[Any],
    output_treedef: Any,
) -> Any:
    positional_specs, named_specs = _selected_input_specs(linear)
    leaf_specs = [
        leaf_spec
        for spec in (*positional_specs, *(spec for _name, spec in named_specs))
        for leaf_spec in spec.leaf_specs
    ]
    columns_by_output: list[list[list[Any]]] = [
        [[] for _leaf_spec in leaf_specs] for _output_leaf in output_leaves
    ]

    seed_entries = (
        (leaf_index, {leaf_spec.node_id: _basis_tangent(leaf_spec.primal, basis_index)})
        for leaf_index, leaf_spec in enumerate(leaf_specs)
        if leaf_spec.node_id is not None
        for basis_index in range(_flat_size(leaf_spec.primal))
    )
    for seed_batch in batched(seed_entries, _PULLBACK_MANY_BATCH_SIZE):
        tangent_outputs = linear._apply_seed_tables_many(  # noqa: SLF001
            tuple(seed_table for _leaf_index, seed_table in seed_batch)
        )
        for (leaf_index, _seed_table), tangent_output in zip(
            seed_batch,
            tangent_outputs,
            strict=True,
        ):
            tangent_leaves, tangent_treedef = tree_flatten(tangent_output)
            if tangent_treedef != output_treedef:
                msg = "jacobian JVP changed output pytree structure between seeds"
                raise RuntimeError(msg)
            for columns, tangent_leaf in zip(
                columns_by_output,
                tangent_leaves,
                strict=True,
            ):
                columns[leaf_index].append(tangent_leaf)

    output_blocks = []
    for output_leaf, columns_by_input in zip(
        output_leaves,
        columns_by_output,
        strict=True,
    ):
        output_shape = _shape(output_leaf)
        flat_blocks = []
        for leaf_spec, columns in zip(leaf_specs, columns_by_input, strict=True):
            if leaf_spec.node_id is None:
                flat_blocks.append(None)
                continue
            input_like = leaf_spec.primal
            input_shape = _shape(input_like)
            if _flat_size(input_like) == 0:
                flat_blocks.append(
                    _empty_jacobian_block(
                        _zeros_like(input_like),
                        output_shape=output_shape,
                        output_like=output_leaf,
                    )
                )
                continue
            block = _stack_jacobian_columns(
                columns,
                output_shape=output_shape,
                input_shape=input_shape,
                output_like=output_leaf,
                input_like=input_like,
            )
            flat_blocks.append(
                _unlift_scalar_array(block) if leaf_spec.restore_python_scalar else block
            )
        output_blocks.append(_format_forward_input_blocks(linear, flat_blocks))
    return tree_unflatten(output_treedef, output_blocks)


def jacobian(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
) -> Callable[..., Any]:
    """Return a shape-preserving dense Jacobian for real pytree inputs and outputs.

    Parameters
    ----------
    f
        Callable to differentiate. Its selected inputs and output may be
        pytrees of real arrays or real Python scalars. A ``StagedProgram`` is
        accepted, but each call still uses a concrete, invocation-local
        linearization.
    argnums
        Positional arguments to differentiate. An integer represents that
        input directly in every output block; a tuple represents the selected
        inputs as a tuple in the given order. ``None`` selects argument zero
        unless ``argnames`` is provided, in which case it selects no
        positional arguments. Negative indices are resolved for each call.
    argnames
        Named arguments to differentiate. Each output leaf contains their
        derivative blocks in a dictionary keyed by name. For an ordinary
        callable, a selected name may be passed positionally or by keyword;
        staged named inputs must be passed by keyword. With both positional
        and named selections, each output leaf contains
        ``(positional_blocks, named_blocks)``.

    Returns
    -------
    Callable
        A concrete-tracing function returning an output-shaped pytree of
        input-shaped derivative blocks. For each output leaf and selected
        input leaf, the dense block shape is
        ``output_leaf.shape + input_leaf.shape``; neither side is flattened.
        Static or untraceable selected input leaves have ``None`` blocks.

    Raises
    ------
    IndexError
        If a positional selection is out of range for the transformed call.
    TypeError
        If a selected input contains an unsupported Python complex scalar.
    ValueError
        If positional selections are duplicated, an argument is selected both
        positionally and by name, a selected name is unavailable, or an input
        or output leaf is complex.
    NoJVPError
        If forward assembly is required through an operation without a JVP
        rule.
    NoVJPError
        If reverse assembly is required through an operation without an
        explicit or structurally derivable transpose rule.
    RuntimeError
        If the provider cannot assemble a dense block or a derivative rule
        changes the expected pytree structure between basis seeds.

    Notes
    -----
    Advect chooses forward or reverse assembly from the traced coordinate
    counts and available rule direction. The temporary ``LinearMap`` is always
    closed before the call returns or raises, releasing retained values and
    primitive residuals.

    A general real-linear complex map needs two complex blocks, or one real
    ``2m x 2n`` block, so a single complex matrix would be ambiguous. Complex
    callers should use ``linearize`` instead.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> jacobian = ad.jacobian(lambda x: x**2)(np.array([1.0, 2.0]))
    >>> jacobian.tolist()
    [[2.0, 0.0], [0.0, 4.0]]
    """
    argnums_tuple, single_argnum = _jacobian_selection(argnums, argnames)

    @functools.wraps(f)
    def jacobian_fn(*args: Any, **kwargs: Any) -> Any:
        value, linear = linearize_call(
            f,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=argnames,
            single_argnum=single_argnum,
        )
        try:
            with _use_array_api_version(linear._trace.array_api_version):  # noqa: SLF001
                output_leaves, output_treedef = tree_flatten(value)
                _validate_real_jacobian(linear, output_leaves)
                input_size = _selected_input_size(linear)
                output_size = sum(_flat_size(leaf) for leaf in output_leaves)
                prefer_forward = not _trace_requires_reverse(linear) and (
                    input_size < output_size
                    or (input_size == output_size and _trace_uses_structural_transpose(linear))
                )
                if input_size > 0 and prefer_forward:
                    return _jacobian_forward(
                        linear,
                        output_leaves=output_leaves,
                        output_treedef=output_treedef,
                    )
                return _jacobian_reverse(
                    linear,
                    output_leaves=output_leaves,
                    output_treedef=output_treedef,
                )
        finally:
            linear.close()

    return jacobian_fn


__all__ = ["LinearMap", "jacobian", "jvp", "linearize"]
