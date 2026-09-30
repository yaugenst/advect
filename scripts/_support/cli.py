"""Argument types shared by the evidence commands."""

from __future__ import annotations

import argparse
import math

_BYTE_UNITS = {
    "gib": 1024**3,
    "gb": 1024**3,
    "mib": 1024**2,
    "mb": 1024**2,
    "kib": 1024,
    "kb": 1024,
    "b": 1,
}


def _invalid(expected: str, value: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(f"expected {expected}, got {value!r}")


def positive_int(value: str) -> int:
    """Parse an integer of at least one."""
    parsed = int(value)
    if parsed < 1:
        raise _invalid("a positive integer", value)
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse an integer of at least zero."""
    parsed = int(value)
    if parsed < 0:
        raise _invalid("a non-negative integer", value)
    return parsed


def positive_float(value: str) -> float:
    """Parse a finite number greater than zero."""
    parsed = float(value)
    if not (math.isfinite(parsed) and parsed > 0):
        raise _invalid("a finite positive number", value)
    return parsed


def fraction(value: str) -> float:
    """Parse a number strictly between zero and one."""
    parsed = float(value)
    if not 0 < parsed < 1:
        raise _invalid("a fraction between zero and one", value)
    return parsed


def byte_size(value: str) -> int:
    """Parse a positive integer byte count or a binary KiB/MiB/GiB size."""
    number = value.strip().lower().replace("_", "")
    multiplier = 1
    for suffix, unit in _BYTE_UNITS.items():
        if number.endswith(suffix):
            number, multiplier = number[: -len(suffix)], unit
            break
    try:
        scaled = float(number) * multiplier
    except ValueError as error:
        raise _invalid("a byte size", value) from error
    if not (math.isfinite(scaled) and scaled >= 1):
        raise _invalid("a finite positive byte size", value)
    return int(scaled)
