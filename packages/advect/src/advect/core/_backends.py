# ruff: noqa: ANN401, SLF001
# ANN401: Public API uses Any for backend-agnostic dispatch
"""Narrow backend registration and input dispatch for Advect."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Input handler registry: list of (accepts, handler) tuples
# accepts: Callable[[Any], bool] - predicate for matching values
# handler: Callable[[Any, str | None], Any] - creates traced value
_input_handlers: list[tuple[Callable[[Any], bool], Callable[..., Any]]] = []
# Exact built-in/backend types can bypass value-dependent predicate dispatch.
# This is deliberately opt-in: arbitrary plugin predicates may depend on the
# value rather than only its type and therefore remain on the ordered fallback.
_exact_input_handlers: dict[type[Any], Callable[..., Any]] = {}


class _BackendState:
    __slots__ = ("core_handlers_loaded",)

    def __init__(self) -> None:
        self.core_handlers_loaded = False


_state = _BackendState()

# Registered hook functions from backends
_hooks: dict[str, Callable[..., Any]] = {}


def register_input_handler(
    accepts: Callable[[Any], bool],
    handler: Callable[..., Any],
    *,
    exact_types: tuple[type[Any], ...] = (),
) -> None:
    """Register an input handler for a backend.

    Parameters
    ----------
    accepts
        Predicate function that returns True if the handler can process
        a given value (e.g., ``lambda v: isinstance(v, np.ndarray)``).
    handler
        Function that creates a traced value from the input. Should have
        signature ``handler(value, name=None) -> traced array``.
    exact_types
        Concrete types for which this handler is unconditionally valid. Exact
        matches bypass ordered predicates; subclasses and value-dependent
        matches still use ``accepts``.

    """
    for exact_type in exact_types:
        existing = _exact_input_handlers.get(exact_type)
        if existing is not None and existing is not handler:
            msg = f"An exact input handler is already registered for {exact_type.__name__}"
            raise ValueError(msg)
        _exact_input_handlers[exact_type] = handler

    for existing_accepts, existing_handler in _input_handlers:
        if existing_accepts is accepts and existing_handler is handler:
            return
    _input_handlers.append((accepts, handler))


def register_hook(name: str, fn: Callable[..., Any]) -> None:
    """Register a single-assignment hook; identical callable registration is idempotent.

    Backends can register callable hooks that the autodiff layer uses
    without importing the backend directly. Common hooks:
    - "<backend>.evaluate_op": Evaluate one backend operation
    - "<backend>.decode_attrs": Decode encoded operation attrs

    Parameters
    ----------
    name
        Hook identifier (e.g., "backend.evaluate_op").
    fn
        The function to register.

    """
    existing = _hooks.get(name)
    if existing is fn:
        return
    if existing is not None:
        msg = f"Hook {name!r} is already registered"
        raise ValueError(msg)
    _hooks[name] = fn


def get_hook(name: str) -> Callable[..., Any] | None:
    """Get a registered hook function.

    Parameters
    ----------
    name
        Hook identifier.

    Returns
    -------
    Callable | None
        The registered function, or None if not registered.
    """
    return _hooks.get(name)


def dispatch_input(
    value: Any,
    name: str | None = None,
    *,
    active: bool = True,
) -> Any:
    """Dispatch input creation to the appropriate backend.

    Tries explicitly registered handlers in order.

    Parameters
    ----------
    value
        The value to create an input from.
    name
        Optional debug name for the input node.

    Returns
    -------
    Any
        A traced array value.

    Raises
    ------
    TypeError
        If no backend can handle the given value type.

    """
    _ensure_core_input_handlers()

    exact_handler = _exact_input_handlers.get(type(value))
    if exact_handler is not None:
        return (
            exact_handler(value, name=name)
            if active
            else exact_handler(value, name=name, active=False)
        )

    # Try registered value-dependent handlers.
    for accepts, handler in _input_handlers:
        if accepts(value):
            return handler(value, name=name) if active else handler(value, name=name, active=False)

    msg = (
        f"No backend can handle input of type {type(value).__name__}. "
        "Pass an array supported by a registered backend."
    )
    raise TypeError(msg)


def _ensure_core_input_handlers() -> None:
    """Install the backend-neutral Array API frontend once per process."""
    if _state.core_handlers_loaded:
        return

    from advect.core._array_api import frontend as _array_api  # noqa: PLC0415

    register_input_handler(_array_api._accepts_array_api, _array_api._handle_array_api_input)
    register_hook("advect.array_api.wrap_traced", _array_api._wrap_traced)

    # Core semantics take precedence even when another provider frontend was
    # imported eagerly before the first dynamic dispatch.
    core_handlers = ((_array_api._accepts_array_api, _array_api._handle_array_api_input),)
    for core_handler in reversed(core_handlers):
        for index, registered in enumerate(_input_handlers):
            if registered[0] is core_handler[0] and registered[1] is core_handler[1]:
                _input_handlers.insert(0, _input_handlers.pop(index))
                break
    _state.core_handlers_loaded = True
