# ruff: noqa: ANN401
# ANN401: Backend registration uses Any for type-agnostic dispatch
"""NumPy dispatch, tracing, and a transparent compatibility namespace.

This package contains the NumPy-specific implementation:
- TracedArray: Wrapper that intercepts NumPy operations
- Ufunc dispatch: Handling for np.add, np.sin, etc.
- Array function dispatch: Handling for np.sum, np.reshape, etc.

Ordinary attributes are returned directly from NumPy. Only constructors that
need to preserve live Advect values are defined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from advect.core._backends import (
    register_hook,
    register_input_handler,
)
from advect.core._context import _get_active_recorder
from advect.numpy._abstract_protocol import (
    _NumpyAbstractArray,
    nested_array_function,
    nested_array_ufunc,
)
from advect.numpy._attrs import decode_attrs
from advect.numpy._constructors import array, asanyarray, asarray
from advect.numpy._eval import (
    bind_evaluator as numpy_bind_evaluator,
    evaluate_op as numpy_evaluate_op,
)
from advect.numpy._traced_array import TracedArray

if TYPE_CHECKING:
    from advect.core._native import DynamicTape


# Register numpy backend with core
def _accepts_numpy(value: Any) -> bool:
    """Accept only NumPy-owned values; other array protocols keep their frontend."""
    return isinstance(value, (np.ndarray, np.generic, TracedArray))


def _handle_numpy_input(
    value: Any,
    name: str | None = None,
    *,
    active: bool = True,
) -> Any:
    """Create a TracedArray from a numpy array."""
    recorder = _get_active_recorder()
    if recorder is None:
        msg = "NumPy inputs require an active trace"
        raise RuntimeError(msg)
    array = value if hasattr(value, "shape") and hasattr(value, "dtype") else np.asarray(value)
    node_id = recorder.record_input(
        array,
        array.shape,
        array.dtype,
        name=name,
        active=active,
    )
    return TracedArray(value=array, node_id=node_id, recorder=recorder, owned=False)


def _wrap_traced(value: Any, *, node_id: int, recorder: DynamicTape) -> Any:
    """Keep primitive results in the NumPy frontend selected by their provider."""
    return TracedArray(value, node_id, recorder)


register_input_handler(
    _accepts_numpy,
    _handle_numpy_input,
    exact_types=(np.ndarray,),
)
register_hook("numpy.evaluate_op", numpy_evaluate_op)
register_hook("numpy.bind_evaluator", numpy_bind_evaluator)
register_hook("numpy.decode_attrs", decode_attrs)
register_hook("numpy.wrap_traced", _wrap_traced)
register_hook("advect.abstract_array_factory", _NumpyAbstractArray)
register_hook("advect.foreign_array_function", nested_array_function)
register_hook("advect.foreign_array_ufunc", nested_array_ufunc)
register_hook("advect.default_array_namespace", lambda: np)


def __getattr__(name: str) -> Any:
    """Return unmodified attributes from the installed NumPy module."""
    return getattr(np, name)


def __dir__() -> list[str]:
    """Expose the real NumPy namespace plus Advect's explicit overrides."""
    return sorted(set(globals()).union(dir(np)))


__all__ = ["array", "asanyarray", "asarray"]
