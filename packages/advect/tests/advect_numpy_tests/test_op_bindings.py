"""Contract tests for NumPy-to-canonical operation names."""

from __future__ import annotations

from advect.numpy._op_bindings import canonicalize_numpy_op, decanonicalize_array_op


def test_operation_names_roundtrip_between_numpy_and_canonical() -> None:
    for suffix in ("add", "cbrt", "fft.fft", "linalg.svd"):
        legacy_op = f"numpy.{suffix}"
        canonical_op = canonicalize_numpy_op(legacy_op)
        assert decanonicalize_array_op(canonical_op) == legacy_op
        assert canonical_op.startswith(("array.", "array_ext."))
