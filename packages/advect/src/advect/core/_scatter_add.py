"""The ``advect.scatter_add`` operation: the transpose of a one-dimensional ``take``.

``scatter_add(values, indices, axis=axis, size=size)`` adds each slice of
``values`` along ``axis`` into zeros of length ``size`` at its entry of the
one-dimensional ``indices``, which lie in ``[0, size)``; repeated indices
accumulate, as with NumPy's unbuffered ``add.at``. Derivative rules emit it
for the adjoint of every gather, so no frontend spells it and one function
serves each lifetime: it records onto the trace of a traced operand and
otherwise evaluates through the operands' provider.
"""

from __future__ import annotations

from math import prod
from typing import Any

from advect.core._abstract import _record_abstract_op
from advect.core._abstract_helpers import normalize_axis
from advect.core._array_api.providers import ResolvedArrayNamespace, _get_array_namespace
from advect.core._context import _select_deepest_active_recorder, get_source_location
from advect.core._primitive_call import _leaf_to_dynamic_operand, _wrap_traced_output
from advect.core._protocols import _is_traced

__all__ = ["scatter_add"]

_OP = "advect.scatter_add"


def _is_abstract(value: object) -> bool:
    return bool(getattr(type(value), "__advect_abstract_array__", False))


def scatter_add(
    values: Any,  # noqa: ANN401 - provider arrays or tracers of any frontend
    indices: Any,  # noqa: ANN401
    *,
    axis: int,
    size: int,
    namespace: Any | None = None,  # noqa: ANN401 - a provider namespace or its resolution
) -> Any:  # noqa: ANN401
    """Scatter-add ``values`` along ``axis`` into ``size`` positions.

    ``namespace`` evaluates concrete operands, which otherwise use their own.
    The evaluation calls only 2022.12 functions, so it needs no revision.
    """
    # A loaded artifact may carry a negative axis; every lifetime sees it normalized.
    axis = normalize_axis(axis, len(values.shape))
    operands = (values, indices)
    dynamic = [operand for operand in operands if _is_traced(operand) and not _is_abstract(operand)]
    if dynamic:
        return _record_dynamic(operands, dynamic, axis=axis, size=size)
    abstract = next((operand for operand in operands if _is_abstract(operand)), None)
    if abstract is not None:
        return _record_abstract_op(abstract._trace, _OP, operands, {"axis": axis, "size": size})  # noqa: SLF001
    resolved = _get_array_namespace(values, api_version=None) if namespace is None else namespace
    if isinstance(resolved, ResolvedArrayNamespace):
        resolved = resolved.raw_namespace
    if resolved is None:
        msg = f"Cannot execute {_OP!r} without an array namespace"
        raise RuntimeError(msg)
    return _evaluate(resolved, values, indices, axis=axis, size=size)


def _record_dynamic(
    operands: tuple[Any, Any],
    dynamic: list[Any],
    *,
    axis: int,
    size: int,
) -> Any:  # noqa: ANN401
    """Record one node on the innermost dynamic trace of the operands.

    Its value scatters the operands' payloads, which records the same node on
    every enclosing trace that still traces them.
    """
    recorder: Any = _select_deepest_active_recorder(operand.recorder for operand in dynamic)
    parents: list[int] = []
    positions: list[int] = []
    literals: list[Any] = []
    payloads: list[Any] = []
    for position, operand in enumerate(operands):
        resolved = _leaf_to_dynamic_operand(recorder, operand)
        if resolved is None:
            msg = f"{_OP!r} operands must be arrays, got {type(operand).__name__}"
            raise TypeError(msg)
        node_id, payload = resolved
        if node_id is None:
            literals.append(payload)
        else:
            parents.append(node_id)
            positions.append(position)
        payloads.append(payload)
    value = scatter_add(*payloads, axis=axis, size=size)
    node_id = recorder.record_operation(
        _OP,
        tuple(parents),
        value,
        {"axis": axis, "size": size},
        tuple(int(dimension) for dimension in value.shape),
        value.dtype,
        input_positions=tuple(positions) if literals else None,
        literals=tuple(literals),
        source_location=get_source_location(),
    )
    return _wrap_traced_output(value, node_id=node_id, recorder=recorder, namespace=None)


def _evaluate(
    namespace: Any,  # noqa: ANN401
    values: Any,  # noqa: ANN401
    indices: Any,  # noqa: ANN401
    *,
    axis: int,
    size: int,
) -> Any:  # noqa: ANN401
    """Scatter concrete operands with the provider's own kernels.

    A provider with NumPy's unbuffered ``add.at`` scatters in one pass. Any
    other provider uses only 2022.12 functions: it sorts the indices and sums
    each run of equal indices by doubling, in ``ceil(log2(len(indices)))``
    passes. Unlike a dense one-hot basis this needs linear memory, and unlike
    differences of a cumulative sum it cannot cancel a small total against a
    large prefix.
    """
    shape = tuple(int(dimension) for dimension in values.shape)
    result_shape = (*shape[:axis], size, *shape[axis + 1 :])
    add_at = getattr(getattr(namespace, "add", None), "at", None)
    if callable(add_at):
        result = namespace.zeros(result_shape, dtype=values.dtype)
        add_at(result, (slice(None),) * axis + (indices,), values)
        return result

    xp = namespace
    count = shape[axis]
    outer, inner = prod(shape[:axis]), prod(shape[axis + 1 :])
    device = getattr(values, "device", None)
    if 0 in (count, size, outer, inner):
        return xp.zeros(result_shape, dtype=values.dtype, device=device)
    order = xp.argsort(indices, stable=True)
    keys = xp.take(indices, order)
    sums = xp.take(xp.reshape(values, (outer, count, inner)), order, axis=1)
    # After the pass with offset ``step`` each entry holds the sum of its run
    # over the last ``2 * step`` sorted positions, so a run's last entry holds
    # its whole total once ``step`` reaches ``count``.
    step = 1
    while step < count:
        same_run = xp.reshape(keys[step:] == keys[:-step], (1, count - step, 1))
        earlier = sums[:, :-step, :]
        later = sums[:, step:, :] + xp.where(same_run, earlier, xp.zeros_like(earlier))
        sums = xp.concat((sums[:, :step, :], later), axis=1)
        step *= 2
    # A stable sort places each target after the equal keys, so its sorted
    # rank less the targets before it counts the keys up to that target: the
    # 2023.12 ``searchsorted(keys, targets, side="right")``.
    targets = xp.arange(size, dtype=keys.dtype, device=device)
    ranks = xp.argsort(xp.argsort(xp.concat((keys, targets)), stable=True))
    ends = ranks[count:] - targets
    starts = xp.concat((xp.zeros((1,), dtype=ends.dtype, device=device), ends[:-1]))
    present = ends > starts
    totals = xp.take(sums, xp.where(present, ends - 1, xp.zeros_like(ends)), axis=1)
    totals = xp.where(xp.reshape(present, (1, size, 1)), totals, xp.zeros_like(totals))
    return xp.reshape(totals, result_shape)
