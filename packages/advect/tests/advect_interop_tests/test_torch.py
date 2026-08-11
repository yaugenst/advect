"""PyTorch bridge qualification."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from advect.interop.torch import wrap  # noqa: E402 - dependency skip precedes adapter import


def test_torch_bridge_preserves_pytree_device_dtype_and_gradients() -> None:
    calls = 0

    def operation(parameters: dict[str, np.ndarray], scale: np.ndarray):
        nonlocal calls
        calls += 1
        field = parameters["field"] * scale
        return {"field": field, "energy": np.sum(field * field)}

    bridged = wrap(operation)
    field = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32, requires_grad=True)
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    result = bridged({"field": field}, scale=scale)

    assert result["field"].device == field.device
    assert result["field"].dtype == torch.float64
    result["energy"].backward()

    assert calls == 1
    assert field.grad is not None
    assert scale.grad is not None
    assert field.grad.dtype == field.dtype
    assert scale.grad.dtype == scale.dtype
    torch.testing.assert_close(field.grad, 2 * field.detach() * scale.detach() ** 2)
    torch.testing.assert_close(
        scale.grad,
        2 * scale.detach() * torch.sum(field.detach().double() ** 2),
    )


def test_torch_bridge_matches_native_complex_gradients() -> None:
    coefficient = 2.0 + 3.0j
    bridged = wrap(lambda value: coefficient * value)
    sample = torch.tensor(1.5 + 2.0j, dtype=torch.complex128, requires_grad=True)
    bridged_output = bridged(sample)
    bridged_gradient = torch.autograd.grad(
        torch.real(bridged_output * torch.conj(bridged_output)),
        sample,
    )[0]

    native_sample = sample.detach().clone().requires_grad_()
    native_output = coefficient * native_sample
    native_gradient = torch.autograd.grad(
        torch.real(native_output * torch.conj(native_output)),
        native_sample,
    )[0]

    torch.testing.assert_close(bridged_gradient, native_gradient)
    torch.testing.assert_close(
        bridged_gradient,
        torch.tensor(39.0 + 52.0j, dtype=torch.complex128),
    )


def test_torch_bridge_matches_nonholomorphic_complex_outputs() -> None:
    bridged = wrap(
        lambda value: (
            np.conjugate(value),
            np.real(value),
            np.imag(value),
            np.abs(value) ** 2,
        )
    )
    sample = torch.tensor(1.5 + 2.0j, dtype=torch.complex128, requires_grad=True)

    def bridged_loss(value):
        conjugate, real, imag, power = bridged(value)
        return torch.real((1.0 + 2.0j) * conjugate) + 0.3 * real - 0.7 * imag + 0.2 * power

    def native_loss(value):
        return (
            torch.real((1.0 + 2.0j) * torch.conj(value))
            + 0.3 * torch.real(value)
            - 0.7 * torch.imag(value)
            + 0.2 * torch.abs(value) ** 2
        )

    bridged_gradient = torch.autograd.grad(bridged_loss(sample), sample)[0]
    native_sample = sample.detach().clone().requires_grad_()
    native_gradient = torch.autograd.grad(native_loss(native_sample), native_sample)[0]
    torch.testing.assert_close(bridged_gradient, native_gradient)


def test_torch_bridge_handles_an_unused_output_and_no_grad_calls() -> None:
    bridged = wrap(lambda value: {"used": value * value, "unused": value + 1})
    sample = torch.tensor([2.0, -3.0], requires_grad=True)
    bridged(sample)["used"].sum().backward()
    torch.testing.assert_close(sample.grad, torch.tensor([4.0, -6.0]))

    with torch.no_grad():
        result = bridged(sample)
    assert not result["used"].requires_grad


def test_torch_bridge_pullback_is_one_shot() -> None:
    bridged = wrap(lambda value: np.sum(value * value))
    sample = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = bridged(sample)
    loss.backward(retain_graph=True)

    with pytest.raises(RuntimeError, match="closed or consumed"):
        loss.backward()


def test_torch_bridge_rejects_integer_input_leaves() -> None:
    bridged = wrap(lambda value: np.sum(value))  # noqa: PLW0108 - bridge boundary
    with pytest.raises(TypeError, match="only floating and complex tensors"):
        bridged(torch.tensor([1, 2], dtype=torch.int64))


def test_torch_bridge_rejects_dtypes_that_cannot_cross_numpy() -> None:
    bridged = wrap(lambda value: np.sum(value))  # noqa: PLW0108 - bridge boundary
    sample = torch.ones(2, dtype=torch.bfloat16, requires_grad=True)

    with pytest.raises(TypeError, match="cannot cross the NumPy bridge"):
        bridged(sample)
