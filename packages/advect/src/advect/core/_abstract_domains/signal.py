"""Abstract registrations and evaluator for one-dimensional signal operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._abstract_helpers import promote_dtype
from advect.core._abstract_model import ArraySpec, rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


RULES: dict[str, AbstractRule] = {
    "array_ext.convolve": rule(
        "signal_1d",
        2,
        positional=("mode",),
        allowed=("mode",),
    ),
    "array_ext.correlate": rule(
        "signal_1d",
        2,
        positional=("mode",),
        allowed=("mode",),
    ),
}


def _signal_1d(
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    if len(specs[0].shape) != 1 or len(specs[1].shape) != 1:
        raise ValueError("convolve/correlate inputs must be one-dimensional")
    left_size = specs[0].shape[0]
    right_size = specs[1].shape[0]
    if left_size == 0 or right_size == 0:
        raise ValueError("convolve/correlate inputs cannot be empty")
    mode = attrs.get("mode", "full")
    if mode == "full":
        size = left_size + right_size - 1
    elif mode == "same":
        size = max(left_size, right_size)
    elif mode == "valid":
        size = max(left_size, right_size) - min(left_size, right_size) + 1
    else:
        raise ValueError("convolve/correlate mode must be full, same, or valid")
    return (ArraySpec((size,), promote_dtype(specs)),)


EVALUATORS: dict[str, ResultEvaluator] = {"signal_1d": _signal_1d}
