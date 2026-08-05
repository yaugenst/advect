"""Compile-time metadata and static Array API creation contracts."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
from numpy.testing import assert_allclose

import advect as ad


def test_static_creation_functions_stage_and_round_trip() -> None:
    sentinel = strict.asarray([0.0], dtype=strict.float32)

    def create(value: object) -> tuple[object, object, object]:
        namespace = value.__array_namespace__()  # type: ignore[attr-defined]
        return (
            namespace.linspace(
                -1.0,
                1.0,
                5,
                dtype=namespace.float32,
            ),
            namespace.fft.fftfreq(
                6,
                d=0.25,
                dtype=namespace.float64,
            ),
            namespace.fft.rfftfreq(
                6,
                d=0.25,
                dtype=namespace.float32,
            ),
        )

    expected = create(sentinel)
    program = ad.stage(
        create,
        specs=(ad.ArraySpec(sentinel.shape, sentinel.dtype),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for candidate in (program(sentinel), restored(sentinel)):
        for actual, reference in zip(candidate, expected, strict=True):
            assert actual.dtype == reference.dtype
            assert_allclose(np.asarray(actual), np.asarray(reference), rtol=1e-7, atol=1e-7)


def test_dtype_metadata_is_resolved_during_abstract_tracing() -> None:
    floating = strict.asarray([0.25, 0.5], dtype=strict.float32)
    integer = strict.asarray([1, 2], dtype=strict.int32)

    def floating_metadata(value: object) -> object:
        namespace = value.__array_namespace__()  # type: ignore[attr-defined]
        info = namespace.finfo(value)
        if not namespace.can_cast(value, namespace.float64):
            return -value  # type: ignore[operator]
        return value + namespace.asarray(info.eps, dtype=value.dtype)  # type: ignore[attr-defined,operator]

    def integer_metadata(value: object) -> object:
        namespace = value.__array_namespace__()  # type: ignore[attr-defined]
        info = namespace.iinfo(value.dtype)  # type: ignore[attr-defined]
        valid = info.bits == 32 and info.max == 2_147_483_647
        return value if valid else -value  # type: ignore[operator]

    for function, value in (
        (floating_metadata, floating),
        (integer_metadata, integer),
    ):
        expected = function(value)
        program = ad.stage(
            function,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )
        restored = ad.StagedProgram.from_dict(program.to_dict())
        assert_allclose(np.asarray(program(value)), np.asarray(expected), rtol=1e-7, atol=1e-7)
        assert_allclose(np.asarray(restored(value)), np.asarray(expected), rtol=1e-7, atol=1e-7)
