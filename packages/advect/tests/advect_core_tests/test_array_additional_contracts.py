"""Additional public contracts for Advect's root array constructors."""

from __future__ import annotations

import importlib.metadata
import runpy
from pathlib import Path
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad


def test_concrete_constructors_honor_provider_dtype_and_copy_forms() -> None:
    default = ad.asarray([1, 2], dtype=np.float32)
    source = np.array([1.0, 2.0])
    borrowed = ad.asarray(source, copy=False)
    owned = ad.array(source)
    strict_source = strict.asarray([1.0, 2.0], dtype=strict.float32)
    strict_converted = ad.asarray(strict_source, dtype=strict.float64, copy=True)

    assert default.dtype == np.dtype(np.float32)
    assert np.shares_memory(borrowed, source)
    assert not np.shares_memory(owned, source)
    assert type(strict_converted) is type(strict_source)
    assert strict_converted.dtype == strict.float64


def test_traced_constructor_handles_constants_and_copy_requests() -> None:
    value = np.array([1.0, 2.0])

    assert_allclose(
        ad.grad(lambda traced: np.sum(ad.array([traced[0], 2.0])))(value),
        np.array([1.0, 0.0]),
    )
    with pytest.raises(ValueError, match="avoid a copy while changing dtype"):
        ad.grad(lambda traced: np.sum(ad.asarray(traced, dtype=np.float32, copy=False)))(value)


def test_asarray_rejects_cyclic_sequences() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)

    with pytest.raises(ValueError, match="does not accept cyclic sequences"):
        ad.asarray(cycle)


def test_asarray_rejects_an_array_namespace_without_asarray() -> None:
    class Value:
        __advect_namespace_is_instance_specific__ = True

        def __array_namespace__(
            self,
            *,
            api_version: str | None = None,
        ) -> object:
            assert api_version == "2024.12"
            return object()

    with pytest.raises(TypeError, match="does not provide asarray"):
        ad.asarray(Value())


def test_source_checkout_has_a_local_version_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_version)

    namespace = runpy.run_path(str(Path(ad.__file__)))

    assert namespace["__version__"] == "0.0.0+local"
