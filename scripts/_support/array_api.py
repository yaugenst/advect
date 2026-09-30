"""Array API helpers shared by the qualification and support-report commands."""

from __future__ import annotations

import inspect


def normalized_signature(function: object) -> str:
    """Render a signature without annotations, as the official stubs are compared."""
    signature = inspect.signature(function)
    return str(
        signature.replace(
            parameters=[
                parameter.replace(annotation=inspect.Parameter.empty)
                for parameter in signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
    )
