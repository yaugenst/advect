"""Concrete NumPy ``__array_function__`` dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from advect.numpy._array_function.algorithms import register_algorithm_handlers
from advect.numpy._array_function.aliases import register_alias_handlers
from advect.numpy._array_function.composite import register_composite_handlers
from advect.numpy._array_function.creation import register_creation_handlers
from advect.numpy._array_function.emission import (
    _make_atleast_handler,
    _make_binary_handler,
    _make_clip_handler,
    _make_interp_handler,
    _make_like_handler,
    _make_multi_input_handler,
    _make_reduction_handler,
    _make_where_handler,
)
from advect.numpy._array_function.fft import register_fft_handlers
from advect.numpy._array_function.linalg import register_linalg_handlers
from advect.numpy._array_function.misc import register_misc_handlers
from advect.numpy._array_function.ordering import register_ordering_handlers
from advect.numpy._array_function.polynomial import register_polynomial_handlers
from advect.numpy._array_function.predicates import register_predicate_handlers
from advect.numpy._array_function.scimath import register_scimath_handlers
from advect.numpy._array_function.shape import register_shape_handlers
from advect.numpy._array_function.signal import register_signal_handlers
from advect.numpy._array_function.split import register_split_handlers
from advect.numpy._array_function.statistics import register_statistics_handlers
from advect.numpy._array_function.unique import register_unique_handlers
from advect.numpy._composite_lowering import REDUCTIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._array_function.emission import ArrayFunctionHandler


def _register_all_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    for name in sorted(REDUCTIONS | {"cumprod", "cumsum"}):
        function = getattr(np, name)
        handlers[function] = _make_reduction_handler(function)
    # concatenate promotes Python scalars weakly; stack coerces them to arrays first.
    for function, weak_scalars in ((np.concatenate, True), (np.stack, False)):
        handlers[function] = _make_multi_input_handler(
            function, f"numpy.{function.__name__}", weak_scalars=weak_scalars
        )
    for function in (np.zeros_like, np.ones_like, np.empty_like):
        handlers[function] = _make_like_handler(function, f"numpy.{function.__name__}")
    for ndim, function in enumerate((np.atleast_1d, np.atleast_2d, np.atleast_3d), start=1):
        handlers[function] = _make_atleast_handler(
            function, f"array.{function.__name__}", target_ndim=ndim
        )
    handlers[np.dot] = _make_binary_handler(np.dot, "numpy.dot")
    handlers[np.where] = _make_where_handler("numpy.where")
    handlers[np.interp] = _make_interp_handler("numpy.interp")
    handlers[np.clip] = _make_clip_handler("numpy.clip")
    for register in (
        register_algorithm_handlers,
        register_alias_handlers,
        register_composite_handlers,
        register_creation_handlers,
        register_fft_handlers,
        register_linalg_handlers,
        register_misc_handlers,
        register_ordering_handlers,
        register_polynomial_handlers,
        register_predicate_handlers,
        register_scimath_handlers,
        register_shape_handlers,
        register_signal_handlers,
        register_split_handlers,
        register_statistics_handlers,
        register_unique_handlers,
    ):
        register(handlers)


_STATIC_ARRAY_FUNCTIONS = frozenset(
    {
        np.can_cast,
        np.common_type,
        np.isrealobj,
        np.ndim,
        np.shape,
        np.size,
    }
)
ARRAY_FUNCTION_HANDLERS: dict[Callable[..., Any], ArrayFunctionHandler] = {}
_register_all_handlers(ARRAY_FUNCTION_HANDLERS)
