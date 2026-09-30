"""Helpers for binding NumPy's live foreign signatures."""

from __future__ import annotations

import functools
import inspect
import operator
from typing import Any, SupportsIndex, cast

from advect.core._errors import TracingError

_ONLY = inspect.Parameter.POSITIONAL_ONLY
_EITHER = inspect.Parameter.POSITIONAL_OR_KEYWORD
_POSITIONAL_KINDS = (_ONLY, _EITHER)
_REQUIRED = inspect.Parameter.empty
# NumPy 2.0 implements these in C without a signature ``inspect`` can read.
# Each entry follows the signature later releases publish: the number of
# leading positional-only parameters, then every positional name and default.
_UNINSPECTABLE_POSITIONAL_PARAMETERS: dict[str, tuple[int, tuple[tuple[str, object], ...]]] = {
    "bincount": (1, (("x", _REQUIRED), ("weights", None), ("minlength", 0))),
    "concatenate": (1, (("arrays", _REQUIRED), ("axis", 0), ("out", None))),
    "dot": (0, (("a", _REQUIRED), ("b", _REQUIRED), ("out", None))),
    "empty_like": (
        1,
        (
            ("prototype", _REQUIRED),
            ("dtype", None),
            ("order", "K"),
            ("subok", True),
            ("shape", None),
        ),
    ),
    "inner": (2, (("a", _REQUIRED), ("b", _REQUIRED))),
    "lexsort": (0, (("keys", _REQUIRED), ("axis", -1))),
    "vdot": (2, (("a", _REQUIRED), ("b", _REQUIRED))),
    "where": (3, (("condition", _REQUIRED), ("x", None), ("y", None))),
}
# NumPy's C implementation binds these by keyword too, although their
# published signatures mark them positional-only.
_KEYWORD_BOUND_POSITIONAL_ONLY = frozenset({"empty_like"})


@functools.lru_cache(maxsize=512)
def positional_parameters(func: object) -> tuple[inspect.Parameter, ...] | None:
    """Return a callable's positional parameters, or ``None`` when unreadable."""
    name = getattr(func, "__name__", "")
    try:
        signature = inspect.signature(cast("Any", func))
    except (TypeError, ValueError):
        known = _UNINSPECTABLE_POSITIONAL_PARAMETERS.get(name)
        if known is None:
            return None
        positional_only, published = known
        parameters = tuple(
            inspect.Parameter(label, _ONLY if index < positional_only else _EITHER, default=default)
            for index, (label, default) in enumerate(published)
        )
    else:
        parameters = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in _POSITIONAL_KINDS
        )
    if name in _KEYWORD_BOUND_POSITIONAL_ONLY:
        return tuple(parameter.replace(kind=_EITHER) for parameter in parameters)
    return parameters


def normalize_required_positionals(
    positional: tuple[inspect.Parameter, ...] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    func: object,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Move required positional-or-keyword operands into handler positions.

    ``positional`` comes from ``positional_parameters``.  NumPy's dispatcher has
    already validated the call against the signature, but the Python dispatcher
    of a C function accepts operands by a keyword the implementation rejects.
    """
    if positional is None:
        return args, kwargs
    by_keyword = [p.name for p in positional[len(args) :] if p.kind is _ONLY and p.name in kwargs]
    if by_keyword:
        msg = (
            f"{getattr(func, '__name__', func)}() got some positional-only arguments "
            f"passed as keyword arguments: '{', '.join(by_keyword)}'"
        )
        raise TypeError(msg)
    normalized = list(args)
    remaining = dict(kwargs)
    for parameter in positional[len(args) :]:
        if (
            parameter.kind is _ONLY
            or parameter.default is not _REQUIRED
            or parameter.name not in remaining
        ):
            break
        normalized.append(remaining.pop(parameter.name))
    return tuple(normalized), remaining


def keyword_optional_positionals(
    positional: tuple[inspect.Parameter, ...] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Pass each optional positional-or-keyword argument by its parameter name.

    Staging reads operands by position and metadata by name, so
    ``np.cumsum(x, 0)`` stages exactly as ``np.cumsum(x, axis=0)`` does.
    Arguments beyond a variadic signature's named parameters stay positional.
    """
    if positional is None or len(args) > len(positional):
        return args, kwargs
    for index, parameter in enumerate(positional[: len(args)]):
        if parameter.kind is _EITHER and parameter.default is not _REQUIRED:
            named = zip((item.name for item in positional[index:]), args[index:], strict=False)
            return args[:index], dict(named) | kwargs
    return args, kwargs


_CLIP_BOUNDS = ("a_min", "a_max")
CLIP_KEYWORDS = frozenset({*_CLIP_BOUNDS, "min", "max"})


def bind_clip_bounds(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return ``clip``'s operand and bounds as NumPy's own ``clip`` binds them.

    NumPy 2.1 gave ``a_min`` and ``a_max`` defaults, so signature normalization
    leaves a keyword ``a_max`` in ``kwargs`` after a positional ``a_min``.
    ``min=``/``max=`` apply only when neither of those is given, and an absent
    bound means no bound.  Every lifetime binds ``clip`` here.
    """
    bounds = dict(zip(_CLIP_BOUNDS, args[1:], strict=False))
    bounds |= {name: kwargs[name] for name in _CLIP_BOUNDS if name in kwargs}
    if not bounds:
        return args[0], kwargs.get("min"), kwargs.get("max")
    if len(bounds) != len(_CLIP_BOUNDS):
        msg = (
            "numpy.clip is only supported during tracing as clip(a, a_min, a_max) "
            "or clip(a, *, min=..., max=...)"
        )
        raise TracingError(msg)
    if "min" in kwargs or "max" in kwargs:
        msg = "numpy.clip does not support mixing min/max with a_min/a_max during tracing"
        raise TracingError(msg)
    return args[0], bounds["a_min"], bounds["a_max"]


def ascending_sort_kwargs(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``sort``/``argsort`` keywords without NumPy's ascending ``descending``.

    NumPy reads ``descending`` by truthiness. The NumPy backend evaluates
    only ascending sorts, so every lifetime rejects a descending one here.
    """
    values = dict(kwargs)
    if values.pop("descending", False):
        msg = f"numpy.{name} kwargs not supported during tracing: ['descending']"
        raise TracingError(msg)
    return values


# NumPy's integer clip modes CLIP, WRAP, and RAISE index this tuple.
_TAKE_MODES = ("clip", "wrap", "raise")


def take_mode(mode: object) -> str:
    """Return ``take``'s index mode by name; every lifetime binds it here.

    NumPy also spells ``raise`` as ``None`` and accepts byte strings and its
    integer clip modes.
    """
    if mode is None:
        return "raise"
    if isinstance(mode, bytes):
        mode = mode.decode("latin-1")
    if isinstance(mode, str) and mode in _TAKE_MODES:
        return str(mode)
    if isinstance(mode, SupportsIndex) and not isinstance(mode, bool):
        number = operator.index(mode)
        if 0 <= number < len(_TAKE_MODES):
            return _TAKE_MODES[number]
    raise TracingError("numpy.take mode must be raise, wrap, or clip")


__all__ = [
    "CLIP_KEYWORDS",
    "ascending_sort_kwargs",
    "bind_clip_bounds",
    "keyword_optional_positionals",
    "normalize_required_positionals",
    "positional_parameters",
    "take_mode",
]
