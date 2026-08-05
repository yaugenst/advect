"""Griffe extension that materializes Advect's lazy autodiff exports.

``advect/__init__.py`` exposes the autodiff transforms through a module
``__getattr__``, which static analysis cannot see. This extension mirrors
that mapping onto the ``advect`` package so ``::: advect.grad`` directives
and cross-references resolve under their public names.
"""

from __future__ import annotations

import griffe


class AdvectLazyExports(griffe.Extension):
    """Alias lazy autodiff exports onto the `advect` package."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        """Add aliases for the lazily exported autodiff names."""
        del kwargs
        if pkg.name != "advect" or "autodiff" not in pkg.members:
            return
        autodiff = pkg["autodiff"]
        api = autodiff["api"]
        targets: dict[str, griffe.Object] = {}
        for member in api.members.values():
            if not isinstance(member, griffe.Module):
                continue
            for export in member.exports or []:
                name = str(export)
                candidate = member[name]
                if name not in targets or not isinstance(candidate, griffe.Alias):
                    targets[name] = candidate
        for name, target in targets.items():
            if name not in pkg.members:
                pkg.set_member(name, griffe.Alias(name, target))
