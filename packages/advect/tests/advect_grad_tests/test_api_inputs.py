"""Tests for autodiff API input-selection helpers."""

from __future__ import annotations

from typing import Never

import pytest

import advect as ad
from advect.autodiff.api import inputs as inputs_api
from advect.autodiff.api.inputs import _get_signature


def test_get_signature_supports_unhashable_callable_instances() -> None:
    class _UnhashableCallable:
        def __call__(self, x: float, y: float = 1.0) -> float:
            return x + y

    type.__setattr__(_UnhashableCallable, "__hash__", None)
    callable_obj = _UnhashableCallable()
    sig = _get_signature(callable_obj)
    assert tuple(sig.parameters) == ("x", "y")


def test_positional_parameter_names_fall_back_for_uninspectable_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_metadata(_callable: object) -> Never:
        msg = "signature unavailable"
        raise ValueError(msg)

    monkeypatch.setattr(inputs_api, "_get_signature_metadata", fail_metadata)

    assert inputs_api._get_positional_param_names(lambda: None) == ((), None)


def test_grad_argnames_still_require_selected_argument_to_be_passed() -> None:
    def f(x: float, y: float = 2.0) -> float:
        return x * y

    grad_f = ad.grad(f, argnums=(), argnames=("y",))

    with pytest.raises(ValueError, match="was not provided in the call"):
        grad_f(3.0)
