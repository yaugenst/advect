"""Canonical operation-name helpers for array-family ops."""

from __future__ import annotations

from functools import lru_cache

# Array API spellings whose canonical array-family leaf has another name;
# every other Array API function shares its canonical leaf name.
ARRAY_API_TO_CANONICAL = {
    "abs": "absolute",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "bitwise_invert": "invert",
    "bitwise_left_shift": "left_shift",
    "bitwise_right_shift": "right_shift",
    "concat": "concatenate",
    "conj": "conjugate",
    "cumulative_prod": "cumprod",
    "cumulative_sum": "cumsum",
    "linalg.cross": "cross",
    "linalg.diagonal": "diagonal",
    "linalg.matmul": "matmul",
    "linalg.matrix_transpose": "transpose",
    "linalg.outer": "outer",
    "linalg.tensordot": "tensordot",
    "linalg.trace": "trace",
    "linalg.vecdot": "vecdot",
    "matrix_transpose": "transpose",
    "permute_dims": "transpose",
    "pow": "power",
    "round": "rint",
}


@lru_cache(maxsize=1)
def _canonical_array_names() -> frozenset[str]:
    # Import lazily so the naming helper remains below the abstract-definition
    # authority without pulling the autodiff rule modules into a root import.
    from advect.core._abstract_domains import operation_semantics  # noqa: PLC0415

    return frozenset(name for name, _schema, _evaluator in operation_semantics())


def _canonical_array_family_op_name(suffix: str) -> str:
    """Resolve one conventional frontend suffix from canonical semantics."""
    candidate = f"array.{suffix}"
    return candidate if candidate in _canonical_array_names() else f"array_ext.{suffix}"
