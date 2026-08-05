"""Contract tests for array-api-strict backend-provider integration."""

from __future__ import annotations

from typing import Any, Self

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose

import advect.autodiff.api.higher_order as higher_order_api
from advect.autodiff.api.common import (
    _allocate_hessian_blocks_flat,
    _prepare_higher_order_inputs,
    _require_array_namespace_for_higher_order,
)
from advect.autodiff.rules.array_family.providers import (
    register_array_family_backend_provider,
    resolve_array_family_backend_provider,
)
from advect.core._errors import HigherOrderNotSupportedError
from advect_grad_tests._backend_contract_helpers import (
    BackendProviderStub,
    StrictHigherOrderNamespace,
)

xp = pytest.importorskip("array_api_strict")


def _register_strict_provider(namespace: Any) -> BackendProviderStub:
    provider = BackendProviderStub(backend="array_api_strict", namespace=namespace)
    register_array_family_backend_provider(provider)
    return provider


def test_array_api_strict_provider_resolves_from_runtime_value(
    isolated_provider_registry: None,
) -> None:
    provider = _register_strict_provider(xp)
    runtime_value = xp.asarray([1.0, 2.0], dtype=xp.float64)

    resolved = resolve_array_family_backend_provider(runtime_value)
    assert resolved is provider


def test_array_api_strict_missing_namespace_capabilities_raise_typed_error(
    isolated_provider_registry: None,
) -> None:
    _register_strict_provider(xp)
    runtime_value = xp.asarray([1.0, 2.0], dtype=xp.float64)

    with pytest.raises(HigherOrderNotSupportedError) as exc_info:
        _ = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    message = str(exc_info.value)
    assert "diag" in message
    assert "zeros_like" in message


def test_array_api_strict_adapter_satisfies_higher_order_namespace_contract(
    isolated_provider_registry: None,
) -> None:
    namespace = StrictHigherOrderNamespace(xp)
    provider = _register_strict_provider(namespace)
    runtime_value = xp.asarray([1.0, -2.0], dtype=xp.float64)

    resolved = resolve_array_family_backend_provider(runtime_value)
    assert resolved is provider
    array_ns = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    assert array_ns is namespace

    _shapes, flat_sizes, primal_dtypes = _prepare_higher_order_inputs(
        array_ns=array_ns,
        args=(runtime_value,),
        argnums=(0,),
    )
    blocks = _allocate_hessian_blocks_flat(
        array_ns=array_ns,
        primal_flat_sizes=flat_sizes,
        primal_dtypes=primal_dtypes,
    )

    assert flat_sizes == [int(runtime_value.size)]
    assert blocks[0][0].shape == (int(runtime_value.size), int(runtime_value.size))


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    arr=hnp.arrays(
        np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=4),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
    )
)
def test_array_api_strict_adapter_preserves_hessian_block_shape_contract(
    isolated_provider_registry: None,
    arr: np.ndarray[Any, Any],
) -> None:
    namespace = StrictHigherOrderNamespace(xp)
    _register_strict_provider(namespace)
    runtime_value = xp.asarray(arr, dtype=xp.float64)

    array_ns = _require_array_namespace_for_higher_order(args=(runtime_value,), kwargs={})
    _shapes, flat_sizes, primal_dtypes = _prepare_higher_order_inputs(
        array_ns=array_ns,
        args=(runtime_value,),
        argnums=(0,),
    )
    blocks = _allocate_hessian_blocks_flat(
        array_ns=array_ns,
        primal_flat_sizes=flat_sizes,
        primal_dtypes=primal_dtypes,
    )

    flat_size = int(runtime_value.size)
    assert flat_sizes == [flat_size]
    assert blocks[0][0].shape == (flat_size, flat_size)
    expected_dtype = array_ns.result_type(primal_dtypes[0], primal_dtypes[0])
    assert blocks[0][0].dtype == expected_dtype


def test_hessian_diag_uses_array_api_strict_adapter_namespace(
    monkeypatch: pytest.MonkeyPatch,
    isolated_provider_registry: None,
) -> None:
    _register_strict_provider(StrictHigherOrderNamespace(xp))

    class IdentityLinearMap:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc_info: object) -> None:
            return None

        def transpose_many(self, cotangents: tuple[object, ...]) -> tuple[object, ...]:
            return cotangents

    def fake_linearize_call(
        _function: object,
        *,
        args: tuple[object, ...],
        **_kwargs: object,
    ) -> tuple[object, IdentityLinearMap]:
        return args[0], IdentityLinearMap()

    monkeypatch.setattr(higher_order_api, "linearize_call", fake_linearize_call)

    runtime_value = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    diag_fn = higher_order_api.hessian_diag(lambda x: x, argnums=0)
    diag = diag_fn(runtime_value)

    assert diag.shape == runtime_value.shape
    assert_allclose(np.asarray(diag), np.ones((3,), dtype=np.float64))
