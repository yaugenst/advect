"""Derivative rules for structural operations owned by Advect's runtime."""

from __future__ import annotations


def _vjp_getoutput(
    _answer: object,
    *,
    g: object,
    index: int,
    num_outputs: int,
    **_attrs: object,
) -> tuple[tuple[object | None, ...]]:
    contributions: list[object | None] = [None] * num_outputs
    contributions[index] = g
    return (tuple(contributions),)


def _jvp_getoutput(
    _answer: object,
    _parent: object,
    *,
    tangents: tuple[object | None, ...],
    index: int,
    num_outputs: int,
    **_attrs: object,
) -> object:
    if len(tangents) != 1:
        msg = "advect.getoutput JVP expects exactly one input tangent"
        raise ValueError(msg)
    parent_tangent = tangents[0]
    if parent_tangent is None:
        return None
    if not isinstance(parent_tangent, tuple):
        msg = "advect.getoutput JVP expects tuple tangent for multi-output parent"
        raise TypeError(msg)
    if len(parent_tangent) != num_outputs:
        msg = (
            "advect.getoutput JVP tangent arity mismatch: "
            f"expected {num_outputs}, got {len(parent_tangent)}"
        )
        raise ValueError(msg)
    if index < 0 or index >= num_outputs:
        msg = f"advect.getoutput JVP index {index} out of range for {num_outputs} outputs"
        raise IndexError(msg)
    return parent_tangent[index]
