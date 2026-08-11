"""Public error-message contracts."""

from __future__ import annotations

from advect.core import NoJVPError, NoVJPError


def test_custom_reverse_error_points_to_primitive_transpose_api() -> None:
    message = str(NoVJPError("missing transpose", op="custom.acme.solve"))

    assert "@primitive_handle.def_transpose" in message
    assert "@primitive_handle.def_jvp" in message
    assert message.index("def_jvp") < message.index("def_transpose")
    assert "flat tuple with one contribution per dynamic input leaf" in message
    assert "check_primitive from advect.testing" in message
    assert "get_primitive" not in message
    assert "get_registry" not in message
    assert "register_vjp" not in message


def test_custom_forward_error_points_to_primitive_jvp_api() -> None:
    message = str(NoJVPError("missing JVP", op="custom.acme.solve"))

    assert "@primitive_handle.def_jvp" in message
    assert "get_primitive" not in message
    assert "get_registry" not in message
    assert "register_jvp" not in message


def test_builtin_derivative_errors_do_not_offer_private_registry_mutation() -> None:
    messages = (
        str(NoVJPError("missing transpose", op="array.sin")),
        str(NoJVPError("missing JVP", op="array.sin")),
    )

    assert all("get_registry" not in message for message in messages)
    assert all("register_" not in message for message in messages)
