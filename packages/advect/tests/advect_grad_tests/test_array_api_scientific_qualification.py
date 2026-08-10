"""Backend-neutral scientific workloads across both Advect lifetimes."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scripts import qualify_array_providers

import advect as ad
from advect.core._array_api.profiles import SUPPORTED_ARRAY_API_VERSIONS


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
@pytest.mark.parametrize("provider_name", ["numpy", "array-api-strict"])
def test_scientific_provider_qualification(
    provider_name: str,
    array_api_version: str,
) -> None:
    provider = qualify_array_providers._providers(
        (provider_name,),
        array_api_version,
    )[0]
    result = qualify_array_providers._qualify_provider(
        provider,
        qualify_array_providers._build_programs(array_api_version),
        array_api_version=array_api_version,
    )

    assert result.report["name"] == provider_name
    assert result.report["selected_array_api_version"] == array_api_version
    assert result.report["arrays"]
    assert "timing_us" not in result.report
    assert "memory_before" not in result.report
    assert "memory_after" not in result.report


def _versioned_update_loss(field: object) -> object:
    namespace = field.__array_namespace__()
    updated = field.copy()
    old_version = updated * updated
    updated[0] = 5.0
    return namespace.sum(old_version + updated)


@pytest.mark.parametrize(
    "field",
    [
        np.asarray([2.0, 3.0], dtype=np.float64),
        strict.asarray([2.0, 3.0], dtype=strict.float64),
    ],
    ids=["numpy", "array-api-strict"],
)
def test_functionalized_update_preserves_prior_ssa_versions(field: object) -> None:
    value, dynamic = ad.value_and_grad(_versioned_update_loss)(field)
    program = ad.stage(
        _versioned_update_loss,
        specs=(ad.ArraySpec(field.shape, field.dtype),),
    )
    staged_value = program(field)
    staged = ad.grad(program)(field)

    assert_allclose(np.asarray(value), 21.0)
    assert_allclose(np.asarray(staged_value), 21.0)
    assert_allclose(np.asarray(dynamic), np.asarray([4.0, 7.0]))
    assert_allclose(np.asarray(staged), np.asarray([4.0, 7.0]))
