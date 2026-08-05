"""Contraction-style linalg VJP exceptions."""

from __future__ import annotations

import math
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _moveaxis, xp
from advect.autodiff.rules.array_family._impl.vjp.exceptions.linalg.common import (
    _EINSUM_ELLIPSIS,
    _EINSUM_FALLBACK_SUBSTRING,
    _MIN_MATRIX_NDIM,
    _h,
    _normalize_axis,
    _normalize_tensordot_axes,
    _shape_of,
)


def _contraction_vjp(
    *,
    a: xp.ndarray,
    b: xp.ndarray,
    g: xp.ndarray,
    a_axes: tuple[int, ...],
    b_axes: tuple[int, ...],
    op_name: str,
    active_input_indices: tuple[int, ...] | None = None,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    active_inputs = (
        frozenset((0, 1)) if active_input_indices is None else frozenset(active_input_indices)
    )
    a_free_axes = tuple(axis for axis in range(a.ndim) if axis not in a_axes)
    b_free_axes = tuple(axis for axis in range(b.ndim) if axis not in b_axes)

    expected_output_shape = tuple(a.shape[axis] for axis in a_free_axes) + tuple(
        b.shape[axis] for axis in b_free_axes
    )
    if tuple(g.shape) != expected_output_shape:
        msg = f"{op_name} expects cotangent shape {expected_output_shape}, got {tuple(g.shape)}"
        raise TypeError(msg)

    contract_shape = tuple(a.shape[axis] for axis in a_axes)
    a_free_shape = tuple(a.shape[axis] for axis in a_free_axes)
    b_free_shape = tuple(b.shape[axis] for axis in b_free_axes)

    a_free_size = math.prod(a_free_shape)
    b_free_size = math.prod(b_free_shape)
    contract_size = math.prod(contract_shape)
    g_mat = xp.reshape(g, (a_free_size, b_free_size))

    grad_a: xp.ndarray | None = None
    if 0 in active_inputs:
        b_perm = b_axes + b_free_axes
        b_mat = xp.reshape(xp.transpose(b, b_perm), (contract_size, b_free_size))
        grad_a_mat = g_mat @ _h(b_mat)
        grad_a_perm = xp.reshape(grad_a_mat, a_free_shape + contract_shape)
        a_perm = a_free_axes + a_axes
        inverse_a_perm = tuple(sorted(range(len(a_perm)), key=a_perm.__getitem__))
        grad_a = xp.transpose(grad_a_perm, inverse_a_perm)

    grad_b: xp.ndarray | None = None
    if 1 in active_inputs:
        a_perm = a_free_axes + a_axes
        a_mat = xp.reshape(xp.transpose(a, a_perm), (a_free_size, contract_size))
        grad_b_mat = _h(a_mat) @ g_mat
        grad_b_perm = xp.reshape(grad_b_mat, contract_shape + b_free_shape)
        b_perm = b_axes + b_free_axes
        inverse_b_perm = tuple(sorted(range(len(b_perm)), key=b_perm.__getitem__))
        grad_b = xp.transpose(grad_b_perm, inverse_b_perm)
    return grad_a, grad_b


def _conjugate_if_complex(value: xp.ndarray) -> xp.ndarray:
    if isinstance(value, complex):
        return cast("xp.ndarray", value.conjugate())
    return cast("xp.ndarray", xp.conj(value)) if xp.iscomplexobj(value) else value


def _vjp_matmul(
    ans: xp.ndarray,
    x: xp.ndarray,
    y: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Apply the real adjoint of ``matmul`` for vectors, matrices, and batches."""
    _ = rest, attrs
    if _shape_of(g) != _shape_of(ans):
        msg = f"numpy.matmul expects cotangent shape {_shape_of(ans)}, got {_shape_of(g)}"
        raise TypeError(msg)

    x_shape = _shape_of(x)
    y_shape = _shape_of(y)
    if not x_shape or not y_shape:
        msg = "numpy.matmul expects inputs with at least one dimension"
        raise ValueError(msg)
    active = frozenset((0, 1)) if active_input_indices is None else frozenset(active_input_indices)
    if not active <= {0, 1}:
        msg = f"numpy.matmul active input indices are invalid: {sorted(active)}"
        raise ValueError(msg)

    x_vector = len(x_shape) == 1
    y_vector = len(y_shape) == 1
    grad_x: xp.ndarray | None = None
    if 0 in active:
        if x_vector and y_vector:
            grad_x = g * _conjugate_if_complex(y)
        elif x_vector:
            product = xp.matmul(xp.expand_dims(g, axis=-2), _h(y))
            grad_x = xp.squeeze(product, axis=-2)
        elif y_vector:
            grad_x = xp.expand_dims(g, axis=-1) * _conjugate_if_complex(y)
        else:
            grad_x = xp.matmul(g, _h(y))

    grad_y: xp.ndarray | None = None
    if 1 in active:
        if x_vector and y_vector:
            grad_y = _conjugate_if_complex(x) * g
        elif x_vector:
            grad_y = xp.expand_dims(_conjugate_if_complex(x), axis=-1) * xp.expand_dims(g, axis=-2)
        elif y_vector:
            if len(x_shape) == _MIN_MATRIX_NDIM:
                grad_y = xp.matmul(_h(x), g)
            else:
                product = xp.matmul(_h(x), xp.expand_dims(g, axis=-1))
                grad_y = xp.squeeze(product, axis=-1)
        else:
            grad_y = xp.matmul(_h(x), g)
    return grad_x, grad_y


def _vjp_dot(
    ans: xp.ndarray | xp.floating[Any] | xp.complexfloating[Any, Any],
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray | xp.floating[Any] | xp.complexfloating[Any, Any],
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """VJP for numpy.dot."""
    _ = rest, attrs
    a_ndim = len(_shape_of(a))
    b_ndim = len(_shape_of(b))

    if _shape_of(g) != _shape_of(ans):
        msg = f"numpy.dot expects cotangent shape {_shape_of(ans)}, got {_shape_of(g)}"
        raise TypeError(msg)

    active_inputs = (
        frozenset((0, 1)) if active_input_indices is None else frozenset(active_input_indices)
    )
    if a_ndim == 0 and b_ndim == 0:
        grad_a = g * xp.conj(b) if 0 in active_inputs else None
        grad_b = g * xp.conj(a) if 1 in active_inputs else None
        return grad_a, grad_b
    if a_ndim == 0:
        grad_a = xp.sum(g * xp.conj(b)) if 0 in active_inputs else None
        grad_b = g * xp.conj(a) if 1 in active_inputs else None
        return grad_a, grad_b
    if b_ndim == 0:
        grad_a = g * xp.conj(b) if 0 in active_inputs else None
        grad_b = xp.sum(xp.conj(a) * g) if 1 in active_inputs else None
        return grad_a, grad_b

    b_contract_axis = (b_ndim - 1,) if b_ndim == 1 else (b_ndim - 2,)
    return _contraction_vjp(
        a=a,
        b=b,
        g=cast("xp.ndarray", g),
        a_axes=(a_ndim - 1,),
        b_axes=b_contract_axis,
        op_name="numpy.dot",
        active_input_indices=active_input_indices,
    )


def _vjp_tensordot(
    ans: xp.ndarray | xp.floating[Any] | xp.complexfloating[Any, Any],
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray | xp.floating[Any] | xp.complexfloating[Any, Any],
    axes: int | tuple[Any, Any] = 2,
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """VJP for numpy.tensordot."""
    _ = rest, attrs

    if _shape_of(g) != _shape_of(ans):
        msg = f"numpy.tensordot expects cotangent shape {_shape_of(ans)}, got {_shape_of(g)}"
        raise TypeError(msg)

    a_axes, b_axes = _normalize_tensordot_axes(
        axes=axes,
        a_shape=_shape_of(a),
        b_shape=_shape_of(b),
        op_name="numpy.tensordot",
    )
    return _contraction_vjp(
        a=a,
        b=b,
        g=cast("xp.ndarray", g),
        a_axes=a_axes,
        b_axes=b_axes,
        op_name="numpy.tensordot",
        active_input_indices=active_input_indices,
    )


def _vjp_vecdot(
    ans: xp.ndarray,
    x1: xp.ndarray,
    x2: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    axis: int = -1,
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Transpose the Array API conjugating vector product."""
    _ = rest, attrs
    if _shape_of(g) != _shape_of(ans):
        msg = f"linalg.vecdot expects cotangent shape {_shape_of(ans)}, got {_shape_of(g)}"
        raise TypeError(msg)
    active = frozenset((0, 1)) if active_input_indices is None else frozenset(active_input_indices)
    x1_shape = _shape_of(x1)
    x2_shape = _shape_of(x2)
    x1_axis = _normalize_axis(axis, ndim=len(x1_shape), op_name="linalg.vecdot")
    x2_axis = _normalize_axis(axis, ndim=len(x2_shape), op_name="linalg.vecdot")
    output_batch_rank = len(_shape_of(ans))
    x1_result_axis = x1_axis + output_batch_rank - (len(x1_shape) - 1)
    x2_result_axis = x2_axis + output_batch_rank - (len(x2_shape) - 1)

    expanded_g = xp.expand_dims(g, axis=-1)
    x1_last = _moveaxis(x1, x1_axis, -1)
    x2_last = _moveaxis(x2, x2_axis, -1)
    grad_x1 = _moveaxis(xp.conj(expanded_g) * x2_last, -1, x1_result_axis) if 0 in active else None
    grad_x2 = _moveaxis(expanded_g * x1_last, -1, x2_result_axis) if 1 in active else None
    return grad_x1, grad_x2


def _normalize_einsum_subscripts(subscripts: str) -> str:
    return "".join(subscripts.split())


def _normalize_einsum_optimize(
    *,
    optimize: bool | str | list[Any] | tuple[Any, ...] | None,
) -> bool | str:
    if optimize is None:
        return False
    if isinstance(optimize, bool | str):
        return optimize
    # Explicit contraction paths are tied to one equation; fallback to planner.
    return True


def _einsum_term_labels(term: str) -> tuple[str, ...]:
    return tuple(ch for ch in term.replace(_EINSUM_ELLIPSIS, ""))


def _einsum_has_repeated_labels(term: str) -> bool:
    labels = _einsum_term_labels(term)
    return len(labels) != len(set(labels))


def _einsum_operand_fast_supported(
    *,
    operand_term: str,
    output_term: str,
    other_terms: tuple[str, ...],
) -> bool:
    if _einsum_has_repeated_labels(operand_term):
        return False
    if _EINSUM_ELLIPSIS not in operand_term and (
        _EINSUM_ELLIPSIS in output_term or any(_EINSUM_ELLIPSIS in term for term in other_terms)
    ):
        return False
    if _EINSUM_ELLIPSIS in operand_term and (
        _EINSUM_ELLIPSIS not in output_term
        and all(_EINSUM_ELLIPSIS not in term for term in other_terms)
    ):
        return False

    output_labels = set(_einsum_term_labels(output_term))
    other_labels: set[str] = set()
    for term in other_terms:
        other_labels.update(_einsum_term_labels(term))

    for label in _einsum_term_labels(operand_term):
        if label in output_labels or label in other_labels:
            continue
        return False
    return True


def _einsum_fast_vjp_plan(
    normalized_subscripts: str,
    arity: int,
) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    if _EINSUM_FALLBACK_SUBSTRING not in normalized_subscripts:
        return None
    lhs_subscripts, output_subscripts = normalized_subscripts.split(
        _EINSUM_FALLBACK_SUBSTRING, maxsplit=1
    )
    operand_terms = tuple(lhs_subscripts.split(","))
    if len(operand_terms) != arity:
        return None

    plans: list[tuple[str, tuple[int, ...]]] = []
    for index, operand_term in enumerate(operand_terms):
        other_indices = tuple(i for i in range(arity) if i != index)
        other_terms = tuple(operand_terms[i] for i in other_indices)
        if not _einsum_operand_fast_supported(
            operand_term=operand_term,
            output_term=output_subscripts,
            other_terms=other_terms,
        ):
            return None
        lhs_terms = [output_subscripts, *other_terms]
        equation = f"{','.join(lhs_terms)}->{operand_term}"
        plans.append((equation, other_indices))
    return tuple(plans)


def _vjp_einsum(
    ans: Any,
    *inputs: Any,
    g: Any,
    subscripts: str,
    optimize: bool | str | list[Any] | tuple[Any, ...] | None = None,
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[Any, ...]:
    _ = attrs
    if _shape_of(g) != _shape_of(ans):
        msg = f"numpy.einsum expects cotangent shape {_shape_of(ans)}, got {_shape_of(g)}"
        raise TypeError(msg)

    normalized_subscripts = _normalize_einsum_subscripts(subscripts)
    fast_plan = _einsum_fast_vjp_plan(normalized_subscripts, len(inputs))
    if fast_plan is None:
        msg = (
            "This einsum equation has no explicit real-adjoint rule. "
            "Rewrite it with matmul/tensordot or define a custom @advect.primitive "
            "transpose."
        )
        raise NotImplementedError(msg)

    optimize_arg = _normalize_einsum_optimize(optimize=optimize)
    active_inputs = (
        frozenset(range(len(inputs)))
        if active_input_indices is None
        else frozenset(active_input_indices)
    )
    gradients: list[Any | None] = []
    for input_index, (equation, other_indices) in enumerate(fast_plan):
        if input_index not in active_inputs:
            gradients.append(None)
            continue
        args: list[Any] = [g]
        args.extend(xp.conjugate(inputs[operand_index]) for operand_index in other_indices)
        gradients.append(xp.einsum(equation, *args, optimize=optimize_arg))
    return tuple(gradients)


for _selective_vjp in (_vjp_matmul, _vjp_dot, _vjp_tensordot, _vjp_einsum):
    cast("Any", _selective_vjp).__advect_vjp_for_input_indices__ = _selective_vjp
