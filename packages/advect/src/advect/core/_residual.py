"""Invocation-local state returned by exact primitive forwards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class PrimitiveResult[R]:
    """A primitive's public output and private same-invocation residual.

    ``output`` is the only value returned to the caller. Advect retains
    ``residual`` for the matching derivative invocation and calls ``release``
    exactly once when that invocation state is discarded. The output must
    remain valid after the residual is released.

    Examples
    --------
    >>> import advect as ad
    >>> result = ad.PrimitiveResult(output=3.0, residual="cached state")
    >>> result.output
    3.0
    """

    output: R
    residual: Any = field(repr=False)
    release: Callable[[Any], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.release is not None and not callable(self.release):
            msg = "PrimitiveResult release must be callable or None"
            raise TypeError(msg)


class _ResidualSlot:
    """Single-owner, idempotently released residual payload."""

    __slots__ = ("_closed", "_payload", "_release")

    def __init__(
        self,
        payload: Any,  # noqa: ANN401 - residuals are intentionally opaque
        release: Callable[[Any], None] | None,
    ) -> None:
        self._payload = payload
        self._release = release
        self._closed = False

    @property
    def payload(self) -> Any:  # noqa: ANN401 - residuals are intentionally opaque
        if self._closed:
            msg = "Primitive residual has already been released"
            raise RuntimeError(msg)
        return self._payload

    def close(self) -> None:
        """Release this payload once, clearing ownership before the callback."""
        if self._closed:
            return
        payload = self._payload
        release = self._release
        self._payload = None
        self._release = None
        self._closed = True
        if release is not None:
            release(payload)

    def __del__(self) -> None:
        """Best-effort leak backstop; deterministic paths call :meth:`close`."""
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors cannot propagate cleanup failures
            return


class _PrimitiveExecution:
    """Internal forward result whose residual ownership can move to a tape."""

    __slots__ = ("_residual", "output")

    def __init__(self, output: Any, residual: _ResidualSlot | None) -> None:  # noqa: ANN401
        self.output = output
        self._residual = residual

    @property
    def residual(self) -> Any | None:  # noqa: ANN401 - residuals are intentionally opaque
        slot = self._residual
        return None if slot is None else slot.payload

    def take_residual(self) -> _ResidualSlot | None:
        """Transfer residual ownership out of this execution."""
        residual = self._residual
        self._residual = None
        return residual

    def close(self) -> None:
        """Release a residual that has not been transferred."""
        residual = self.take_residual()
        if residual is not None:
            residual.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors cannot propagate cleanup failures
            return


def _normalize_primitive_execution(
    value: Any,  # noqa: ANN401 - primitive results are backend-generic
    *,
    primitive_name: str,
    has_residual: bool,
) -> _PrimitiveExecution:
    """Validate one implementation result against its residual contract."""
    if has_residual:
        if not isinstance(value, PrimitiveResult):
            msg = (
                f"Primitive '{primitive_name}' declares residual=True; its implementation "
                "must return PrimitiveResult(output, residual, release=...)"
            )
            raise TypeError(msg)
        return _PrimitiveExecution(
            value.output,
            _ResidualSlot(value.residual, value.release),
        )
    if isinstance(value, PrimitiveResult):
        residual = _ResidualSlot(value.residual, value.release)
        residual.close()
        msg = (
            f"Primitive '{primitive_name}' returned PrimitiveResult but does not declare "
            "residual=True"
        )
        raise TypeError(msg)
    return _PrimitiveExecution(value, None)


__all__ = ["PrimitiveResult"]
